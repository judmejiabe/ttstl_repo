import pandas as pd

df = pd.read_parquet("runs_market_data_test/2025-09-09_INTC_bdelta7200000000000_adelta600000000000_sdelta5000000000_seed19991810_idd8b30d/bb_000/global_update/backbone/stationarity/stationarity_summary.parquet")
df.to_csv("stationarity.csv", index=True)