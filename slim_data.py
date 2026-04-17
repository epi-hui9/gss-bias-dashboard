import pyreadstat, pandas as pd, pickle, os

DTA = "/Users/epi_hui9/epi_hui9/Work/NU Courses/Spring 2026/MSAI 490/GSS_stata/gss7224_r3.dta"
print("Loading full GSS…")
df, meta = pyreadstat.read_dta(DTA, encoding="latin1")

cat = pd.read_parquet("catalog.parquet")
keep = ["year"] + [v for v in cat[cat["n_years"] >= 10]["var"].tolist() if v in df.columns]

slim = df[keep].copy()
slim.to_parquet("gss_slim.parquet", compression="zstd")

labels = {
    "variable_value_labels": {v: meta.variable_value_labels.get(v, {}) for v in keep},
    "column_names_to_labels": {v: meta.column_names_to_labels.get(v, "") for v in keep},
}
with open("gss_labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print(f"slim shape: {slim.shape}")
print(f"gss_slim.parquet: {os.path.getsize('gss_slim.parquet')/1e6:.1f} MB")
print(f"gss_labels.pkl:   {os.path.getsize('gss_labels.pkl')/1e6:.1f} MB")
