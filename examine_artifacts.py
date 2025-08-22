import pandas as pd

df = pd.read_parquet("runs/2025-08-22_SIM_bdelta500000_adelta100000_sdelta100_seed19991810_id61f662/bb_000/global_update/interface/variable_selection_info.parquet")
df.to_csv("variable_selection_info.csv", index=True)