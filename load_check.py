import pyreadstat
DTA = "/Users/epi_hui9/epi_hui9/Work/NU Courses/Spring 2026/MSAI 490/GSS_stata/gss7224_r3.dta"
print(f"Loading {DTA} ...")
df, meta = pyreadstat.read_dta(DTA, encoding="latin1")
print("Shape:", df.shape)
print("Years covered:", sorted([int(y) for y in df["year"].dropna().unique()]))
print("First 15 columns:", list(df.columns[:15]))
