import pandas as pd

df = pd.read_parquet("runs/2025-08-28_SIM_bdelta500000_adelta100000_sdelta100_seed19991810_id657603/bb_001/local_adapt/adapter/performance.parquet")
df.to_csv("performance.csv", index=True)