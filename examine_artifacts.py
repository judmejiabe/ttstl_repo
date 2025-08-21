import pandas as pd

df = pd.read_parquet("runs/2025-08-20_SIM_bdelta500000_adelta100000_sdelta100_seed19991810_id546c99/bb_000/global_update/backbone/stationarity/stationarity_summary.parquet")
df.to_csv("stationarity.csv", index=True)