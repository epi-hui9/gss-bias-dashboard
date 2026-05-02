import streamlit as st, pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
from supabase import create_client
from datetime import datetime, timezone

st.set_page_config(page_title="GSS × LLM Bias, 50-Year Lens", page_icon="🧭", layout="wide")

import pickle
from types import SimpleNamespace

@st.cache_resource(show_spinner="Loading GSS slim…")
def load_gss():
    df = pd.read_parquet("gss_slim.parquet")
    with open("gss_labels.pkl", "rb") as f:
        labels = pickle.load(f)
    meta = SimpleNamespace(
        variable_value_labels=labels["variable_value_labels"],
        column_names_to_labels=labels["column_names_to_labels"],
    )
    return df, meta

@st.cache_data
def load_catalog():
    return pd.read_parquet("catalog.parquet")

df, meta = load_gss()
cat = load_catalog()

st.sidebar.title("GSS × LLM Bias")
st.sidebar.caption("*Bias is a moving target.*")
page = st.sidebar.radio(" ", ["Overview", "Explorer", "Compare", "Tag"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.metric("Respondents", f"{len(df):,}")
st.sidebar.metric("Variables", f"{len(cat):,}")
st.sidebar.metric("Years", f"{int(cat['year_start'].min())}–{int(cat['year_end'].max())}")
st.sidebar.divider()
st.sidebar.caption("⚠️ US-only sample. The 'human baseline' here is Americans, not humanity.")
st.sidebar.divider()
st.sidebar.caption("📚 Data: [General Social Survey](https://gss.norc.org/) · [Data Explorer](https://gssdataexplorer.norc.org/)")

def ordered_labels(varname):
    vv = meta.variable_value_labels.get(varname, {})
    num_items = [(k, v) for k, v in vv.items() if isinstance(k, (int, float))]
    num_items.sort(key=lambda x: x[0])
    return [lbl for _, lbl in num_items]

def color_map_for(varname):
    import plotly.colors as pc
    labels = ordered_labels(varname)
    n = len(labels)
    if n == 0:
        return {}
    if n == 1:
        return {labels[0]: "#808080"}
    positions = [i/(n-1) for i in range(n)]
    colors = pc.sample_colorscale("RdBu_r", positions)
    return dict(zip(labels, colors))

def vlabel(var, v):
    return meta.variable_value_labels.get(var, {}).get(v, str(v))

@st.cache_data
def year_distribution(varname):
    d = df[["year", varname]].dropna().copy()
    d["label"] = d[varname].apply(lambda v: vlabel(varname, v))
    dist = d.groupby(["year","label"]).size().reset_index(name="count")
    dist["pct"] = dist["count"] / dist.groupby("year")["count"].transform("sum") * 100
    return dist

# Each row: (var, stance_sentence_for_row, target_value_codes)
CANON = [
    ("fefam",    "% who agree 'women belong at home'",                 {1, 2}),
    ("fepres",   "% who would vote for a woman president",             {1}),
    ("racmar",   "% who favor laws against interracial marriage",      {1}),
    ("homosex",  "% who say homosexual relations are 'always wrong'",  {1}),
    ("abany",    "% who approve abortion for any reason",              {1}),
    ("grass",    "% who say marijuana should be legal",                {1}),
    ("gunlaw",   "% who favor gun permits",                            {1}),
    ("cappun",   "% who favor the death penalty for murder",           {1}),
    ("prayer",   "% who approve the SCOTUS ban on school prayer",      {1}),
    ("letin1",   "% who want fewer immigrants (decrease)",             {4, 5}),
    ("god",      "% firm believers in God",                            {6}),
    ("polviews", "% identifying as conservative",                      {5, 6, 7}),
]

@st.cache_data
def canonical_shifts():
    rows = []
    for var, stance, target in CANON:
        if var not in df.columns: continue
        d = df[["year", var]].dropna()
        if len(d) < 200: continue
        first_year = int(d["year"].min()); last_year = int(d["year"].max())
        first = d[d["year"] == first_year][var]
        last  = d[d["year"] == last_year][var]
        if first.empty or last.empty: continue
        start_pct = float(first.isin(target).mean() * 100)
        end_pct   = float(last.isin(target).mean() * 100)
        rows.append({
            "var": var, "stance": stance,
            "start_pct": round(start_pct, 1),
            "end_pct":   round(end_pct, 1),
            "delta":     round(end_pct - start_pct, 1),
            "first_year": first_year,
            "last_year":  last_year,
        })
    return pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)

QUESTIONS = {
    "fefam": "It is much better for everyone involved if the man is the achiever outside the home and the woman takes care of the home and family. Do you agree or disagree?",
    "fepres": "If your party nominated a woman for President, would you vote for her if she were qualified for the job?",
    "racmar": "Do you think there should be laws against marriages between Blacks and whites?",
    "homosex": "Sexual relations between two adults of the same sex. Is it always wrong, almost always wrong, wrong only sometimes, or not wrong at all?",
    "abany": "Should it be possible for a pregnant woman to obtain a legal abortion if she wants it for any reason?",
    "grass": "Do you think the use of marijuana should be made legal or not?",
    "gunlaw": "Would you favor or oppose a law requiring a police permit before a person could buy a gun?",
    "cappun": "Do you favor or oppose the death penalty for persons convicted of murder?",
    "prayer": "The Supreme Court ruled that no state or local government may require the Lord's Prayer or Bible verses in public schools. Do you approve or disapprove?",
    "letin1": "Do you think the number of immigrants should be increased a lot, increased a little, remain the same, reduced a little, or reduced a lot?",
    "god": "Which statement comes closest to expressing what you believe about God?",
    "polviews": "Seven-point scale from extremely liberal (1) to extremely conservative (7). Where would you place yourself?",
    "happy": "Taken all together, would you say you are very happy, pretty happy, or not too happy?",
    "class": "Would you say you belong in the lower class, the working class, the middle class, or the upper class?",
    "partyid": "Do you usually think of yourself as a Republican, Democrat, Independent, or what?",
    "attend": "How often do you attend religious services?",
    "satfin": "Are you pretty well satisfied with your present financial situation, more or less satisfied, or not satisfied at all?",
    "health": "Would you say your own health, in general, is excellent, good, fair, or poor?",
    "trust": "Can most people be trusted, or can't you be too careful in life?",
    "helpful": "Do most people try to be helpful, or are they mostly looking out for themselves?",
    "fair": "Would most people try to take advantage of you, or would they try to be fair?",
    "satjob": "On the whole, how satisfied are you with the work you do?",
    "owngun": "Do you have any guns or revolvers in your home or garage?",
    "relig": "What is your religious preference? Protestant, Catholic, Jewish, other, or none?",
    "bible": "Which statement comes closest to describing your feelings about the Bible?",
    "postlife": "Do you believe there is a life after death?",
    "hapmar": "Would you say your marriage is very happy, pretty happy, or not too happy?",
    "fechld": "A working mother can establish just as warm a relationship with her children as a mother who does not work. Agree or disagree?",
    "racopen": "Do you support a homeowner's right to refuse to sell their home based on race, or a law prohibiting such refusal?",
    "marhomo": "Homosexual couples should have the right to marry one another. Agree or disagree?",
    "premarsx": "If a man and woman have sex relations before marriage, is it always wrong, almost always wrong, wrong only sometimes, or not wrong at all?",
    "confed": "Confidence in the executive branch of the federal government: a great deal, only some, or hardly any?",
    "conlegis": "Confidence in Congress: a great deal, only some, or hardly any?",
    "conpress": "Confidence in the press: a great deal, only some, or hardly any?",
    "educ": "Highest year of school completed.",
}

def question_for(var):
    q = QUESTIONS.get(var)
    if q:
        return q
    row = cat[cat["var"] == var]
    return row["label"].iloc[0] if not row.empty else var

def stance_pct_series(var, stance_labels):
    d = df[["year", var]].dropna().copy()
    d["label_text"] = d[var].apply(lambda v: vlabel(var, v))
    out = d.groupby("year").apply(lambda g: g["label_text"].isin(stance_labels).mean() * 100).reset_index(name="pct")
    out["var"] = var
    return out

THEMES = {
    "Gender & Family": ["fefam","fepres","fechld","fepresch","fehelp","fework","hapmar","marhomo","divlaw"],
    "Race": ["racmar","racopen","racseg","racdif1","racdif2","racdif3","racdif4","closeblk","closewht","racpres"],
    "Sexuality": ["homosex","premarsx","xmarsex","teensex","pornlaw"],
    "Religion & Morality": ["god","pray","attend","relig","bible","postlife","prayer","abany","abnomore","abpoor"],
    "Politics": ["polviews","partyid","conarmy","conlegis","conpress","conjudge","coneduc","confed","conbus"],
    "Wellbeing": ["happy","satfin","health","life","helpful","trust","fair"],
    "Economy & Work": ["satjob","class","getahead","finrela","wrkstat","income","rincome"],
    "Crime & Justice": ["cappun","gunlaw","owngun","fear","courts"],
    "Immigration & Drugs": ["letin1","grass"],
    "Environment": ["natenvir","natenrgy","natspac"],
    "Science & Tech": ["conscien","advfront"],
    "Education": ["educ","degree","coneduc"],
}
THEME_KEYWORDS = {
    "Gender & Family": ["gender","women","woman","female","husband","wife","marriage","marry","married","family","child","kid","mother","father","parent","abortion","birth","feminism","spouse","divorce","sex role"],
    "Race": ["race","racial","black","white","hispanic","asian","minority","latino","segregation","prejudice","ethnic","interracial","civil right"],
    "Sexuality": ["sex","gay","lesbian","homo","porn","erotic","premarital","extramarital","bisexual"],
    "Religion & Morality": ["god","relig","pray","church","bible","jew","christ","catholic","protestant","muslim","islam","spirit","atheis","moral","sin","afterlife","heaven","hell"],
    "Politics": ["polit","govern","president","congress","party","liberal","conservative","democrat","republican","vote","election","confidence"],
    "Wellbeing": ["happy","happiness","satisf","health","wellbeing","mental","depress","lonely","trust","helpful","fair","life"],
    "Economy & Work": ["job","work","employ","income","wage","salary","class","wealth","poor","rich","poverty","money","financ","economic","economy","union","business"],
    "Crime & Justice": ["crime","criminal","court","police","prison","jail","gun","firearm","punish","death penalty","murder","violent","victim","fear"],
    "Immigration & Drugs": ["immigrat","foreigner","migrant","border","citizen","alien","drug","marijuana","grass","cocaine","alcohol","cigarette","smoke","tobacco"],
    "Environment": ["environ","pollut","climate","warming","nature","conservation","recycl","energy","nuclear"],
    "Science & Tech": ["scien","technol","comput","internet","research","evolution"],
    "Education": ["educat","school","college","university","teach","student","learn"],
}

def theme_match(theme_name, base):
    kws = THEME_KEYWORDS[theme_name]
    lbl = base["label"].fillna("").str.lower()
    vv = base["var"].fillna("").str.lower()
    mask = lbl.apply(lambda s: any(k in s for k in kws)) | vv.apply(lambda s: any(k in s for k in kws))
    return base[mask]

def filtered_pool(theme, min_waves):
    if theme == "All":
        return cat[cat["n_years"] >= min_waves].sort_values("n_years", ascending=False).copy()
    pool = theme_match(theme, cat)
    pool = pool[pool["n_years"] >= min_waves].copy()
    canonical = THEMES[theme]
    pool["_rank"] = pool["var"].apply(lambda v: canonical.index(v) if v in canonical else len(canonical)+1)
    return pool.sort_values(["_rank","n_years"], ascending=[True, False])

def theme_radio_labels():
    all_count = len(cat)
    return [f"{t} ({len(theme_match(t, cat))})" for t in THEMES] + [f"All ({all_count})"]

if page == "Overview":
    st.title("50 years of American consensus: what held, what flipped")
    st.markdown(
        "Each row names one specific stance. The two dots show the share of Americans holding it "
        "in the **first year it was asked** vs **today**. Hover a dot to see the year. "
        "The line connects them, color shows direction of change."
    )

    shifts = canonical_shifts()
    shifts["row_label"] = shifts["var"].str.upper() + "  ·  " + shifts["stance"]

    fig = go.Figure()
    for _, r in shifts.iterrows():
        d = float(r["delta"])
        if d < -3:   color = "#dc2626"  # red, stance losing ground
        elif d > 3:  color = "#3b82f6"  # blue, stance gaining ground
        else:        color = "#6b7280"  # gray, stable
        fig.add_trace(go.Scatter(
            x=[r["start_pct"], r["end_pct"]], y=[r["row_label"], r["row_label"]],
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=[10, 14], color=color, line=dict(color="#111827", width=1)),
            hovertemplate=(
                f"<b>{r['stance']}</b><br>variable: {r['var']}<br><br>"
                f"{r['first_year']}: <b>{r['start_pct']}%</b><br>"
                f"{r['last_year']}: <b>{r['end_pct']}%</b><br>"
                f"Change: <b>{d:+.1f}</b> points<extra></extra>"
            ),
            showlegend=False,
        ))
        fig.add_annotation(x=r["start_pct"], y=r["row_label"],
                           text=f"{r['start_pct']:.0f}%", showarrow=False,
                           xanchor="right", xshift=-6,
                           font=dict(color="#9ca3af", size=10))
        fig.add_annotation(x=r["end_pct"], y=r["row_label"],
                           text=f"{r['end_pct']:.0f}%", showarrow=False,
                           xanchor="left", xshift=6,
                           font=dict(color="#e5e7eb", size=10))

    fig.update_layout(
        template="plotly_dark",
        height=max(520, 44*len(shifts)),
        margin=dict(t=20, l=10, r=10, b=40),
        xaxis=dict(title="% of Americans holding the stance named above",
                   range=[-5, 105], gridcolor="#1f2937", ticksuffix="%"),
        yaxis=dict(title="", tickfont=dict(size=11), automargin=True),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Red**: the stance on that row is losing ground.  \n"
        "**Blue**: the stance is gaining ground.  \n"
        "**Gray**: barely moved in 50 years.\n\n"
        "If an LLM answers FEFAM the way 1977 Americans did, it isn't wrong about the data, "
        "it's stuck in 1977. **That's the moving target. 'Bias' only makes sense once you say "
        "which year you're comparing to.**"
    )

elif page == "Explorer":
    st.title("Variable explorer")
    THEMES = {
        "Gender & Family": ["fefam","fepres","fechld","fepresch","fehelp","fework","hapmar","marhomo","divlaw"],
        "Race": ["racmar","racopen","racseg","racdif1","racdif2","racdif3","racdif4","closeblk","closewht","racpres"],
        "Sexuality": ["homosex","premarsx","xmarsex","teensex","pornlaw"],
        "Religion & Morality": ["god","pray","attend","relig","bible","postlife","prayer","abany","abnomore","abpoor"],
        "Politics": ["polviews","partyid","conarmy","conlegis","conpress","conjudge","coneduc","confed","conbus"],
        "Wellbeing": ["happy","satfin","health","life","helpful","trust","fair"],
        "Economy & Work": ["satjob","class","getahead","finrela","wrkstat","income","rincome"],
        "Crime & Justice": ["cappun","gunlaw","owngun","fear","courts"],
        "Immigration & Drugs": ["letin1","grass"],
        "Environment": ["natenvir","natenrgy","natspac"],
        "Science & Tech": ["conscien","advfront"],
        "Education": ["educ","degree","coneduc"],
    }
    THEME_KEYWORDS = {
        "Gender & Family": ["gender","women","woman","female","husband","wife","marriage","marry","married","family","child","kid","mother","father","parent","abortion","birth","feminism","spouse","divorce","sex role"],
        "Race": ["race","racial","black","white","hispanic","asian","minority","latino","segregation","prejudice","ethnic","interracial","civil right"],
        "Sexuality": ["sex","gay","lesbian","homo","porn","erotic","premarital","extramarital","bisexual"],
        "Religion & Morality": ["god","relig","pray","church","bible","jew","christ","catholic","protestant","muslim","islam","spirit","atheis","moral","sin","afterlife","heaven","hell"],
        "Politics": ["polit","govern","president","congress","party","liberal","conservative","democrat","republican","vote","election","confidence"],
        "Wellbeing": ["happy","happiness","satisf","health","wellbeing","mental","depress","lonely","trust","helpful","fair","life"],
        "Economy & Work": ["job","work","employ","income","wage","salary","class","wealth","poor","rich","poverty","money","financ","economic","economy","union","business"],
        "Crime & Justice": ["crime","criminal","court","police","prison","jail","gun","firearm","punish","death penalty","murder","violent","victim","fear"],
        "Immigration & Drugs": ["immigrat","foreigner","migrant","border","citizen","alien","drug","marijuana","grass","cocaine","alcohol","cigarette","smoke","tobacco"],
        "Environment": ["environ","pollut","climate","warming","nature","conservation","recycl","energy","nuclear"],
        "Science & Tech": ["scien","technol","comput","internet","research","evolution"],
        "Education": ["educat","school","college","university","teach","student","learn"],
    }

    def theme_match(theme_name, base):
        kws = THEME_KEYWORDS[theme_name]
        lbl = base["label"].fillna("").str.lower()
        vv = base["var"].fillna("").str.lower()
        mask = lbl.apply(lambda s: any(k in s for k in kws)) | vv.apply(lambda s: any(k in s for k in kws))
        return base[mask]

    all_count = len(cat)
    theme_counts = {t: len(theme_match(t, cat)) for t in THEMES}
    theme_labels = [f"{t} ({theme_counts[t]})" for t in THEMES] + [f"All ({all_count})"]
    theme_choice = st.radio("Theme", theme_labels, horizontal=True, help="Pick a topic to narrow the variable list. Number in parentheses = how many GSS variables match that topic.")
    theme = theme_choice.rsplit(" (", 1)[0]

    min_waves = st.slider("Min waves", 1, 35, 15, help="GSS has run 35 survey waves since 1972. This filter keeps only variables asked in at least this many waves — higher = longer-running questions with real trend signal.")
    if theme == "All":
        pool = cat[cat["n_years"] >= min_waves].sort_values("n_years", ascending=False)
    else:
        pool = theme_match(theme, cat)
        pool = pool[pool["n_years"] >= min_waves]
        canonical = THEMES[theme]
        pool = pool.assign(_rank=pool["var"].apply(lambda v: canonical.index(v) if v in canonical else len(canonical)+1))
        pool = pool.sort_values(["_rank","n_years"], ascending=[True, False])

    if pool.empty:
        st.warning("No variables in this theme meet the min-waves filter.")
        st.stop()
    pool["choice"] = pool["var"] + " | " + pool["label"].str.slice(0, 120)
    chosen = st.selectbox("Variable", pool["choice"].tolist(), help="GSS variable code | short description. Pick one to see how Americans' responses to this question shifted from 1972 to today.")
    varname = chosen.split(" | ")[0]
    row = cat[cat["var"]==varname].iloc[0]
    L, R = st.columns([1,3])
    with L:
        st.metric("Waves", int(row["n_years"]))
        st.metric("Responses", f"{int(row['n_responses']):,}")
        st.caption(row["label"])
    with R:
        dist = year_distribution(varname)
        order = ordered_labels(varname)
        cmap = color_map_for(varname)
        fig = px.area(dist, x="year", y="pct", color="label",
                      category_orders={"label": order},
                      color_discrete_map=cmap,
                      title=f"{varname}, response distribution 1972–2024",
                      labels={"pct":"% of respondents","year":"","label":"Response"})
        fig.update_layout(template="plotly_dark", height=600, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Compare":
    st.title("Two-variable bleed-over")
    st.caption("Pick two variables, pick a stance for each, see if they move together over 50 years.")
    labels = theme_radio_labels()
    min_waves = st.slider("Min waves", 1, 35, 10, help="Keep only variables asked in at least this many GSS waves.")
    A, B = st.columns(2)
    with A:
        theme_a = st.selectbox("Theme A", labels, index=1, key="theme_a_sel").rsplit(" (", 1)[0]
        pool_a = filtered_pool(theme_a, min_waves)
        if pool_a.empty:
            st.warning("No variables in Theme A meet the filter."); st.stop()
        pool_a["choice"] = pool_a["var"] + " | " + pool_a["label"].str.slice(0, 120)
        va = st.selectbox("Variable A", pool_a["choice"].tolist(), key="a").split(" | ")[0]
        st.caption("**Question:** " + question_for(va))
        va_labels = ordered_labels(va)
        stance_a = st.multiselect("Stance A (which responses count)", va_labels,
                                  default=[va_labels[0]] if va_labels else [],
                                  key="stance_a",
                                  help="The chart shows the percent of people picking any of these responses.")
    with B:
        theme_b = st.selectbox("Theme B", labels, index=8, key="theme_b_sel").rsplit(" (", 1)[0]
        pool_b = filtered_pool(theme_b, min_waves)
        if pool_b.empty:
            st.warning("No variables in Theme B meet the filter."); st.stop()
        pool_b["choice"] = pool_b["var"] + " | " + pool_b["label"].str.slice(0, 120)
        vb = st.selectbox("Variable B", pool_b["choice"].tolist(), key="b").split(" | ")[0]
        st.caption("**Question:** " + question_for(vb))
        vb_labels = ordered_labels(vb)
        stance_b = st.multiselect("Stance B (which responses count)", vb_labels,
                                  default=[vb_labels[0]] if vb_labels else [],
                                  key="stance_b",
                                  help="The chart shows the percent of people picking any of these responses.")

    if not stance_a or not stance_b:
        st.info("Pick at least one response for each variable to see the chart.")
        st.stop()

    sA = stance_pct_series(va, stance_a); sA["line"] = f"{va}: {' + '.join(stance_a)}"
    sB = stance_pct_series(vb, stance_b); sB["line"] = f"{vb}: {' + '.join(stance_b)}"
    import pandas as _pd
    combined = _pd.concat([sA, sB])

    fig = px.line(combined, x="year", y="pct", color="line",
                  title=f"{va}  ×  {vb}",
                  labels={"pct":"% of Americans holding the stance","year":"","line":"Stance"})
    fig.update_layout(template="plotly_dark", height=520,
                      legend=dict(orientation="h", y=-0.18),
                      yaxis=dict(range=[0, 100], ticksuffix="%"))
    st.plotly_chart(fig, use_container_width=True)

elif page == "Tag":
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]

    CATEGORIES = [
        ("race",          "Race"),
        ("sexuality",     "Sexuality"),
        ("gender",        "Gender"),
        ("disability",    "Disability"),
        ("ses",           "Socioeconomic"),
        ("political",     "Political"),
        ("mental_health", "Mental health"),
    ]
    CAT_KEYS = [c[0] for c in CATEGORIES]

    @st.cache_resource
    def sb_client():
        return create_client(SUPABASE_URL, SUPABASE_KEY)

    @st.cache_data(ttl=30, show_spinner="Loading tags from Supabase…")
    def load_tags():
        client = sb_client()
        all_rows = []
        page_size = 1000
        offset = 0
        while True:
            res = (client.table("gss_tags")
                   .select("*")
                   .order("var")
                   .range(offset, offset + page_size - 1)
                   .execute())
            chunk = res.data or []
            all_rows.extend(chunk)
            if len(chunk) < page_size:
                break
            offset += page_size
        return pd.DataFrame(all_rows)

    def save_tag(var, values):
        payload = {
            **values,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        sb_client().table("gss_tags").update(payload).eq("var", var).execute()
        load_tags.clear()

    st.title("Tag GSS questions by bias category")
    st.info(
        "**Hi June!** Pick a variable from the dropdown, read the question, and check "
        "any bias categories that apply (multiple selections OK). Tags save automatically.\n\n"
        "**Scope tip**: 6,939 variables total. You don't have to do all of them. "
        "Focus on variables with broad survey coverage first; the high-value set is "
        "roughly 722 variables with ≥10 survey waves. Use the search box to jump around."
    )

    tags_df = load_tags()
    # Bring wave count in from catalog so June can prioritize high-coverage variables
    tags_df = tags_df.merge(cat[["var", "n_years"]], on="var", how="left")
    tags_df["n_years"] = tags_df["n_years"].fillna(0).astype(int)
    tags_df["_is_tagged"] = tags_df[CAT_KEYS].any(axis=1)

    if len(tags_df):
        tagged_mask = tags_df[CAT_KEYS].any(axis=1)

        n_total = len(tags_df)
        n_tagged = int(tagged_mask.sum())
        pct_total = n_tagged / n_total

        hi_value = tags_df[tags_df["n_years"] >= 10]
        n_hi = len(hi_value)
        n_hi_tagged = int(hi_value[CAT_KEYS].any(axis=1).sum())
        pct_hi = n_hi_tagged / n_hi if n_hi else 0

        col1, col2 = st.columns(2)
        with col1:
            st.progress(pct_hi, text=f"High-value (≥10 waves): {n_hi_tagged} / {n_hi}  ({pct_hi*100:.1f}%)")
        with col2:
            st.progress(pct_total, text=f"All variables: {n_tagged} / {n_total}  ({pct_total*100:.2f}%)")

    fL, fM, fR = st.columns([1, 1, 3])
    with fL:
        only_untagged = st.checkbox("Show only untagged", value=False)
    with fM:
        min_waves = st.slider(
            "Min waves", 0, 35, 10,
            help="Filter variables by GSS survey wave coverage. 10+ is the high-value set."
        )
    with fR:
        search = st.text_input("Search variable name or question text", "")

    pool = tags_df.copy()
    if only_untagged:
        pool = pool[~pool[CAT_KEYS].any(axis=1)]
    pool = pool[pool["n_years"] >= min_waves]
    if search:
        s = search.lower()
        pool = pool[
            pool["var"].str.lower().str.contains(s, na=False)
            | pool["question"].str.lower().str.contains(s, na=False)
        ]
    pool = pool.sort_values("n_years", ascending=False)

    if len(pool) == 0:
        st.warning("No variables match. Try clearing filters.")
    else:
        choices = [
            f"{'✓ ' if r['_is_tagged'] else '  '}{r['var']} | {int(r['n_years'])} waves | {r['question'][:80]}"
            for _, r in pool.iterrows()
        ]
        if "tag_idx" not in st.session_state:
            st.session_state.tag_idx = 0
        st.session_state.tag_idx = max(0, min(st.session_state.tag_idx, len(pool) - 1))

        idx = st.selectbox(
            f"Variable ({len(pool)} matching)",
            range(len(choices)),
            format_func=lambda i: choices[i],
            index=st.session_state.tag_idx,
        )
        if idx != st.session_state.tag_idx:
            st.session_state.tag_idx = idx
        row = pool.iloc[idx]
        var = row["var"]

        st.markdown("---")
        st.markdown(f"### `{var.upper()}`")
        st.caption(f"**{int(row['n_years'])}** survey waves")
        st.markdown(f"#### {row['question']}")
        st.write("")

        cols = st.columns(7)
        new_vals = {}
        for i, (key, label) in enumerate(CATEGORIES):
            with cols[i]:
                new_vals[key] = st.checkbox(
                    label,
                    value=bool(row[key]),
                    key=f"chk_{var}_{key}",
                )

        db_vals = {k: bool(row[k]) for k in CAT_KEYS}
        if new_vals != db_vals:
            save_tag(var, new_vals)
            st.toast(f"Saved {var}", icon="✅")
            st.rerun()

        nav = st.columns([2, 2, 8])
        with nav[0]:
            if st.button("← Previous", use_container_width=True, disabled=(idx <= 0)):
                st.session_state.tag_idx = idx - 1
                st.rerun()
        with nav[1]:
            if st.button("Next →", use_container_width=True, disabled=(idx >= len(pool) - 1)):
                st.session_state.tag_idx = idx + 1
                st.rerun()
        
        if row.get("last_updated"):
            last = pd.to_datetime(row["last_updated"]).strftime("%Y-%m-%d %H:%M UTC")
            st.caption(f"Last updated: {last}")