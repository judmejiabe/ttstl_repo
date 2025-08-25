import pandas as pd

df = pd.read_parquet("runs/2025-08-25_SIM_bdelta500000_adelta100000_sdelta100_seed19991810_id52dd63/bb_001/global_update/adapter/param_series.parquet")
df.to_csv("param_series.csv", index=True)