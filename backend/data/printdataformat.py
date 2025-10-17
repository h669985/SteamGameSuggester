import pandas as pd
# Helper file to dump the data format from the dataset

path = "hf://datasets/FronkonGames/steam-games-dataset/data/train-00000-of-00001-e2ed184370a06932.parquet"
df = pd.read_parquet(path)

print("=== COLUMN NAMES ===")
for c in sorted(df.columns):
    print(c)

print("\n=== DTYPES ===")
print(df.dtypes.sort_index())

print("\n=== SAMPLE ROW ===")
print(df.head(1).to_dict(orient="records")[0])
