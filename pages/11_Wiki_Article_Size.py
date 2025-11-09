import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Streamlit setup ---
st.set_page_config(page_title="Wikipedia Article Size Dashboard", layout="wide")
st.title("🌐 Wikipedia Article Size by Language Edition")

# --------------------------------------------------------------------
# 1. Load DataFrame from session_state
# --------------------------------------------------------------------

STUDENT_DATA_KEY = 'st11_df'

if 'student_data' not in st.session_state:
    st.session_state['student_data'] = {}

if STUDENT_DATA_KEY not in st.session_state['student_data']:
    data_path = os.path.join('data', STUDENT_DATA_KEY.replace('_df','_data') + '.csv')
    if os.path.exists(data_path):
        st.session_state['student_data'][STUDENT_DATA_KEY] = pd.read_csv(data_path)
        st.success(f"✅ Loaded data from {data_path}")
    else:
        st.error(f"❌ Data file not found at {data_path}")
        st.session_state['student_data'][STUDENT_DATA_KEY] = pd.DataFrame()

df = st.session_state['student_data'][STUDENT_DATA_KEY].copy()
if df.empty:
    st.stop()

# --------------------------------------------------------------------
# 2. Clean and prepare data
# --------------------------------------------------------------------
def clean_data(df):
    df = df.copy()
    df["language_edition"] = df["original_url"].str.split(".wikipedia.org").str[0].str.split("//").str[-1]
    df["page_title"] = df["original_url"].str.split("/wiki/").str[-1]
    return df

df_clean = clean_data(df)

df_clean["count"] = df_clean.groupby("language_edition")["language_edition"].transform("count")

english_count = (
    df_clean[df_clean["language_edition"] == "en"]["count"].iloc[0]
    if "en" in df_clean["language_edition"].values
    else 0
)

df_clean["pct_of_en"] = (df_clean["count"] / english_count * 100) if english_count > 0 else 0
df_clean["pct_label"] = df_clean["pct_of_en"].apply(lambda x: f"{x:.1f}%")

lang_map = {
    'en': 'English', 'ar': 'Arabic', 'fr': 'French', 'es': 'Spanish', 'de': 'German',
    'pt': 'Portuguese', 'zh': 'Chinese', 'ru': 'Russian', 'uk': 'Ukrainian', 'it': 'Italian',
    'ja': 'Japanese', 'nl': 'Dutch', 'id': 'Indonesian', 'pl': 'Polish', 'sv': 'Swedish',
    'fi': 'Finnish', 'cs': 'Czech', 'ko': 'Korean', 'he': 'Hebrew', 'el': 'Greek',
    'da': 'Danish', 'hu': 'Hungarian', 'hi': 'Hindi', 'ro': 'Romanian', 'bg': 'Bulgarian'
}
df_clean["language_edition"] = df_clean["language_edition"].map(lang_map).fillna(df_clean["language_edition"])

df_clean["label"] = (
    df_clean["language_edition"] + "\n" + df_clean["pct_label"] +
    "\n(n=" + df_clean["count"].astype(str) + ")"
)

adjust_langs = ['Chinese', 'Japanese', 'Hindi', 'Korean', 'Arabic', 'Russian', 'Ukrainian', 'Hebrew', 'Greek']
df_clean["page_size_new"] = np.where(
    df_clean["language_edition"].isin(adjust_langs),
    df_clean["page_size"] / 2,
    df_clean["page_size"]
)

order = (
    df_clean.groupby("language_edition")["count"]
    .max()
    .sort_values(ascending=False)
    .index.tolist()
)

# --------------------------------------------------------------------
# 4. Plot the graph
# --------------------------------------------------------------------
st.sidebar.header("⚙️ Plot Settings")

metric = "page_size_new" 
show_points = st.sidebar.checkbox("Show individual article points", value=True)

# Create custom x labels with n=
df_clean["label"] = df_clean["language_edition"] + " (n=" + df_clean["count"].astype(str) + ")"

fig = px.box(
    df_clean,
    x="label",
    y=metric,
    points="all" if show_points else "outliers",
    hover_data=["page_title", "language_edition"],
    category_orders={"label": df_clean.sort_values("count", ascending=False)["label"].unique()},
    title="Wikipedia Article Size by Language Edition",
)

fig.update_traces(marker=dict(opacity=0.1, size=1))

fig.update_layout(
    showlegend=False,
    xaxis_title="Language Edition",
    yaxis_title="Page Size (bytes)",
    height=650,
    margin=dict(l=40, r=40, t=60, b=120),
    xaxis=dict(tickangle=45)
)

st.plotly_chart(fig, use_container_width=True)
