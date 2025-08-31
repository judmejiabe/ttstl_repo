import pandas as pd

df = pd.read_parquet("runs/2025-08-29_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_id5f2812/bb_000/local_adapt/adapter/performance.parquet")
df.to_csv("performance_3.csv", index=True)