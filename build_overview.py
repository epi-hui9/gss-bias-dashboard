import pyreadstat, pandas as pd, plotly.express as px

DTA = "/Users/epi_hui9/epi_hui9/Work/NU Courses/Spring 2026/MSAI 490/GSS_stata/gss7224_r3.dta"
print("Loading GSS ...")
df, meta = pyreadstat.read_dta(DTA, encoding="latin1")

print(f"Cataloging {df.shape[1]-1} variables ...")
records = []
for col in df.columns:
    if col == "year": continue
    mask = df[col].notna()
    if not mask.any(): continue
    years = sorted(df.loc[mask, "year"].dropna().unique().astype(int).tolist())
    records.append({
        "var": col,
        "label": meta.column_names_to_labels.get(col, "") or "",
        "n_years": len(years),
        "n_responses": int(mask.sum()),
        "year_start": years[0],
        "year_end": years[-1],
    })

cat = pd.DataFrame(records)

def bucket(n):
    if n >= 30: return "Long-running (30+ waves)"
    if n >= 20: return "Core (20–29 waves)"
    if n >= 10: return "Mid (10–19 waves)"
    if n >= 5:  return "Periodic (5–9 waves)"
    return "One-off (<5 waves)"
cat["coverage_bucket"] = cat["n_years"].apply(bucket)

cat.to_parquet("catalog.parquet")
print(cat["coverage_bucket"].value_counts())

fig = px.treemap(
    cat,
    path=["coverage_bucket", "var"],
    values="n_responses",
    hover_data=["label", "n_years", "year_start", "year_end"],
    color="n_years",
    color_continuous_scale="RdBu_r",
    title=f"GSS — {len(cat)} variables · 1972–2024 · sized by respondent count · colored by # waves covered",
)
fig.update_layout(template="plotly_dark", margin=dict(t=60, l=10, r=10, b=10), font=dict(family="Inter, system-ui"))
fig.write_html("overview.html", include_plotlyjs="cdn")
print("Wrote catalog.parquet + overview.html")
