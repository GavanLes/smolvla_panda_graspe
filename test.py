import pandas as pd

# ★★★ 最关键：不要让 pandas 截断显示 ★★★
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", "{:.6f}".format)

# ★ 修改成你的 parquet 路径 ★
df = pd.read_parquet('./demo_data/data/chunk-000/episode_000001.parquet')

# ★ 打印前 10 行，完整显示数组，不会有“...” ★
print(df[['observation.state','action']].head(10))

#demo_data