import pandas as pd

df_SA_02 = pd.read_parquet("runs_synthetic_study/2025-09-10_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idSA_02/bb_001/local_adapt/adapter/performance.parquet")
df_SA_05 = pd.read_parquet("runs_synthetic_study/2025-09-10_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idSA_05/bb_001/local_adapt/adapter/performance.parquet")



df = pd.merge(df_SA_02, df_SA_05, left_index=True, right_index=True, suffixes=('_SA_02', '_SA_05'))
print(df[['relmae_SA_02', 'relmae_SA_05']])

df = pd.read_parquet("runs_market_data_experiments/2025-09-18_INTC_bdelta7200000000000_adelta600000000000_sdelta500000000_seed19991810_idNO_COVARIATES_ARMA01_PAIR01_05312024_06032024/bb_000/local_adapt/adapter/performance.parquet")
print(df)