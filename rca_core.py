"""
rca_core.py - the RCA pipeline functions, moved out of 02_rca_llm.ipynb so the
Streamlit dashboard (app.py) can use them.

The code is the SAME logic as the notebook (frozen prompt, same context builder,
same anonymisation, same cache key), so cached answers in llm_cache.json match.

Quick self-test (checks the rebuilt contexts match the cached ones):
    python rca_core.py
"""
import json
import re
import time
from typing import List, Literal

import pandas as pd
from pydantic import BaseModel

# ---------------------------------------------------------------- settings
MODEL = "gemini-3.5-flash-lite"
CACHE_FILE = "llm_cache.json"            # research cache (gold + practice), read-only in the dashboard
LIVE_CACHE_FILE = "llm_cache_live.json"  # answers from the dashboard's "Analyse now" button
EVENTS_FILE = "events_slim.parquet"
INCIDENTS_FILE = "incidents_v2.parquet"

ACCOUNT_ID = "811596193553"
LEAK_WORDS = ["flaws", "level5", "level6", ACCOUNT_ID]

SHOW_COLS_ALL = ["eventTime", "eventName", "eventSource", "sourceIPAddress",
                 "userIdentityarn", "errorCode", "userAgent"]
KEY_COLS_ALL = ["eventName", "eventSource", "sourceIPAddress", "userIdentityarn", "errorCode"]


# ---------------------------------------------------------------- 1b. anonymisation
def make_arn_map(all_arns):
    """Real ARN -> anonymised ARN. Built over the sorted unique ARNs of the
    whole event table, exactly like the notebook, so the codes (role-12,
    name-4, ...) are identical."""
    roles, names, sessions = {}, {}, {}

    def anon_arn(a):
        a = str(a)
        if a == "None":
            return a
        out = a.replace(ACCOUNT_ID, "123456789012")
        m = re.match(r"^(.*:assumed-role/)([^/]+)/(.+)$", out)
        if m:
            role = roles.setdefault(m.group(2), f"role-{len(roles) + 1}")
            s = m.group(3)
            sess = sessions.setdefault(
                s, ("ec2-instance-" if s.startswith("i-") else "session-") + str(len(sessions) + 1))
            return m.group(1) + role + "/" + sess
        m = re.match(r"^(.*:(?:user|role)/)(.+)$", out)
        if m:
            return m.group(1) + names.setdefault(m.group(2), f"name-{len(names) + 1}")
        return out

    unique = sorted(pd.Series(all_arns).astype(str).unique())
    return {a: anon_arn(a) for a in unique}


def anonymise(table, arn_map):
    """Keep the real ARN in arn_real (dashboard only), replace userIdentityarn
    with the anonymised one. Safe to call twice."""
    if "arn_real" not in table.columns:
        table = table.copy()
        table["arn_real"] = table["userIdentityarn"].astype(str)
        table["userIdentityarn"] = table["arn_real"].map(arn_map)
    return table


def leak_check(text):
    """Returns a list of real names found in the text (empty list = OK)."""
    return [w for w in LEAK_WORDS if w in text.lower()]


# ---------------------------------------------------------------- loading
def load_events(path=EVENTS_FILE):
    """Load the full event table and anonymise it (slow: ~1.2M rows)."""
    df = pd.read_parquet(path)
    df["eventTime"] = pd.to_datetime(df["eventTime"], errors="coerce")
    arn_map = make_arn_map(df["userIdentityarn"])
    return anonymise(df, arn_map), arn_map


def load_arn_map(path=EVENTS_FILE):
    """Only reads the ARN column, so it is fast. Same map as load_events."""
    arns = pd.read_parquet(path, columns=["userIdentityarn"])["userIdentityarn"]
    return make_arn_map(arns)


# ---------------------------------------------------------------- 2. get_context
def collapse(rows):
    show = [c for c in SHOW_COLS_ALL if c in rows.columns]
    keys_cols = [c for c in KEY_COLS_ALL if c in rows.columns]
    r = rows[show].copy()
    if "userAgent" in r.columns:
        r["userAgent"] = r["userAgent"].astype(str).str[:40]
    keys = r[keys_cols].astype(str)
    block = (keys != keys.shift()).any(axis=1).cumsum()
    agg = {c: (c, "first") for c in show}
    agg["count"] = ("eventName", "size")
    return r.groupby(block).agg(**agg).reset_index(drop=True)


def head_tail(t, n):
    return t if len(t) <= n else pd.concat([t.head(n // 2), t.tail(n - n // 2)])


def list_counts(s):
    return ", ".join(f"{k} x{v}" for k, v in s.items()) or "none"


def get_context(incident, df, window_hours=1, core_rows=20, around_rows=10):
    h, arn, ip = incident["hour"], incident["userIdentityarn"], incident["sourceIPAddress"]
    start, end = h - pd.Timedelta(hours=window_hours), h + pd.Timedelta(hours=1 + window_hours)
    in_time = (df["eventTime"] >= start) & (df["eventTime"] < end)
    ctx = df.loc[in_time & ((df["userIdentityarn"] == arn) | (df["sourceIPAddress"] == ip))].sort_values("eventTime")

    in_hour = (ctx["eventTime"] >= h) & (ctx["eventTime"] < h + pd.Timedelta(hours=1))
    core = ctx[in_hour & (ctx["userIdentityarn"] == arn) & (ctx["sourceIPAddress"] == ip)]
    around = ctx.drop(core.index)

    other_ids = ctx.loc[ctx["sourceIPAddress"] == ip, "userIdentityarn"].value_counts().drop(arn, errors="ignore").head(5)
    other_ips = ctx.loc[ctx["userIdentityarn"] == arn, "sourceIPAddress"].value_counts().drop(ip, errors="ignore").head(5)

    lines = [
        f"INCIDENT: identity={arn} | ip={ip} | hour={h} | identity type={incident['identity_type']} | flagged events={incident['n_events']}",
        f"CONTEXT WINDOW: {start} to {end} | related events={len(ctx)}",
        "TOP EVENTS IN WINDOW: " + list_counts(ctx["eventName"].value_counts().head(10)),
        "OTHER IDENTITIES USING THIS IP: " + list_counts(other_ids),
        "OTHER IPs USED BY THIS IDENTITY: " + list_counts(other_ips),
    ]
    if "errorCode" in ctx.columns:
        lines.append("ERRORS: " + list_counts(ctx.loc[ctx["errorCode"] != "None", "errorCode"].value_counts().head(5)))
    if len(core):
        core_c = head_tail(collapse(core), core_rows)
        lines.append(f"\nINCIDENT-HOUR EVENTS ({len(core)} events, {len(core_c)} rows shown, repeats merged):")
        lines.append(core_c.to_string(index=False))
    if len(around):
        ar_c = head_tail(collapse(around), around_rows)
        lines.append(f"\nSURROUNDING EVENTS ({len(around)} events, {len(ar_c)} rows shown, repeats merged):")
        lines.append(ar_c.to_string(index=False))
    return "\n".join(lines)


# ---------------------------------------------------------------- 2b. history + build_context
def history_lines(incident, df):
    arn, ip, h = incident["userIdentityarn"], incident["sourceIPAddress"], incident["hour"]
    same_id = df["userIdentityarn"] == arn
    same_ip = df["sourceIPAddress"] == ip
    before = df["eventTime"] < h

    id_prior = df.loc[same_id & before, "eventTime"]
    pair_prior = df.loc[same_id & same_ip & before, "eventTime"]
    id_txt = (f"this identity was first seen {id_prior.min():%Y-%m-%d} and was active on {id_prior.dt.date.nunique()} earlier days"
              if len(id_prior) else "this identity has NOT been seen before (first appearance)")
    pair_txt = (f"it used this IP on {pair_prior.dt.date.nunique()} earlier days (first {pair_prior.min():%Y-%m-%d})"
                if len(pair_prior) else "it has NEVER used this IP before")

    in_hour = (df["eventTime"] >= h) & (df["eventTime"] < h + pd.Timedelta(hours=1))
    ua = df.loc[same_id & same_ip & in_hour, "userAgent"].astype(str).str.lower()
    console = ua.str.contains("console")
    script = ~console & ua.str.contains("boto|aws-cli|aws-sdk|botocore|python|terraform|curl")
    internal = ~console & ~script & ua.str.contains("aws-internal|amazonaws.com|awscloudtrail")
    other = ~(console | script | internal)
    return [
        f"HISTORY: {id_txt}; {pair_txt}.",
        f"ACCESS METHOD (incident hour): web console x{int(console.sum())}, CLI/SDK script x{int(script.sum())}, "
        f"AWS internal service x{int(internal.sum())}, other x{int(other.sum())}",
    ]


def build_context(incident, df):
    lines = get_context(incident, df).split("\n")
    return "\n".join(lines[:1] + history_lines(incident, df) + lines[1:])


def build_minimal(incident):
    """Baseline (no_context): only what the detector alert itself shows."""
    return (f"INCIDENT: identity={incident['userIdentityarn']} | ip={incident['sourceIPAddress']} | "
            f"hour={incident['hour']} | identity type={incident['identity_type']} | "
            f"flagged events={incident['n_events']} | most common event={incident['top_event']} | "
            f"error rate={incident['error_rate']:.0%}")


# ---------------------------------------------------------------- 4. prompt + result format (FROZEN)
class RCAResult(BaseModel):
    verdict: Literal["malicious", "suspicious", "benign"]
    root_cause: str
    severity: Literal["low", "medium", "high", "critical"]
    confidence: float
    evidence: List[str]
    next_action: str


INSTRUCTIONS = """You are a cloud security analyst reviewing AWS CloudTrail activity.
An anomaly detector flagged the incident below. Anomaly detectors flag rare activity, not necessarily
malicious activity, so many flags are normal admin work or AWS services.
Names and account numbers have been replaced with codes.
The HISTORY and ACCESS METHOD lines show what is normal for this identity; use them to judge
whether the activity fits its usual behaviour.

Using ONLY the evidence shown:
- verdict: malicious, suspicious, or benign
- root_cause: 1-2 sentences explaining what most likely happened and why it was flagged
- severity: low, medium, high, or critical
- confidence: a number from 0 to 1
- evidence: 2-5 short facts from the log that support your answer
- next_action: one concrete step an analyst should take next
If the evidence is not enough, say so in root_cause and lower your confidence.

"""


# ---------------------------------------------------------------- cache
def cache_key(incident, variant):
    return f"{MODEL}|{variant}|{incident['arn_real']}|{incident['sourceIPAddress']}|{incident['hour']}"


def load_cache(path=CACHE_FILE):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_cache(cache, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


# ---------------------------------------------------------------- analyse
def make_client(api_key):
    from google import genai
    return genai.Client(api_key=api_key)


def analyse_live(incident, df, client, variant="with_context"):
    """One fresh Gemini call (never reads the cache). Returns an entry in the
    same format as llm_cache.json. Raises an error if the context leaks names."""
    t0 = time.perf_counter()
    context = build_context(incident, df) if variant == "with_context" else build_minimal(incident)
    t1 = time.perf_counter()
    leaks = leak_check(context)
    if leaks:
        raise ValueError(f"Leak check failed, not sent to Gemini: {leaks}")
    resp = client.models.generate_content(
        model=MODEL,
        contents=INSTRUCTIONS + context,
        config={"response_mime_type": "application/json", "response_schema": RCAResult, "temperature": 0},
    )
    t2 = time.perf_counter()
    return {
        "result": RCAResult.model_validate_json(resp.text).model_dump(),
        "context": context, "model": MODEL, "variant": variant,
        "context_seconds": round(t1 - t0, 3), "llm_seconds": round(t2 - t1, 3),
        "label_category": incident["label_category"],   # for the record only, NOT sent to Gemini
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    print("Loading events (this takes a little while)...")
    df, _ = load_events()
    gold = pd.read_parquet("gold_labelled.parquet")
    cache = load_cache()
    same = diff = missing = 0
    for _, g in gold.iterrows():
        for variant in ["with_context", "no_context"]:
            entry = cache.get(cache_key(g, variant))
            if entry is None:
                missing += 1
                continue
            text = build_context(g, df) if variant == "with_context" else build_minimal(g)
            if text == entry["context"]:
                same += 1
            else:
                diff += 1
                print("DIFFERENT:", g["gold_id"], variant)
    print(f"\nContexts identical to the cache: {same} | different: {diff} | not in cache: {missing}")
    print("rca_core.py OK" if diff == 0 and missing == 0 else "Check the lines above")
