import pandas as pd

df = pd.read_parquet("runs/2025-08-21_SIM_bdelta500000_adelta100000_sdelta100_seed19991810_id655b77/bb_000/global_update/backbone/architecture/train_log.parquet")
df.to_csv("train_log.csv", index=True)