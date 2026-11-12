import os
import pandas as pd

COLS = ["eventTime", "eventName", "eventSource", "sourceIPAddress",
        "userIdentityarn", "errorCode", "userAgent"]

df = pd.read_parquet("labelled_v2.parquet", columns=COLS)
df.to_parquet("events_slim.parquet", compression="zstd")  # keeps the same row order and index

print("Rows:", len(df))
print("Size:", round(os.path.getsize("events_slim.parquet") / 1024 / 1024, 1), "MB")