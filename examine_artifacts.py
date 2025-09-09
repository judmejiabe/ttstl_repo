import pandas as pd

df = pd.read_parquet("runs_synthetic_study/2025-09-04_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idPL3_03/bb_000/local_adapt/adapter/performance.parquet")
df.to_csv("performance_0.csv", index=True)