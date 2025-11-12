import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import plotly.express as px
import plotly.graph_objects as go

# --- Streamlit setup ---
st.set_page_config(page_title="Figure 3b — Beeswarm", layout="wide")
st.title("Beeswarm of Pageviews per Capita (log)")

# --- Intro ---
st.header("1. Introduction and Project Goal")
st.markdown("""
**Data Description:** We analyze country-level Wikipedia pageviews using a single merged dataset.
This file joins daily country pageview counts with annual country attributes (population, total pageviews, region),
so each row contains a country's date, daily pageviews, and the corresponding regional and population context for that year.

**Research Question:** Beeswarm plot of per‑capita pageviews of all country‑year pairs with continuous traffic.
In our preprocessing, “continuous traffic” is approximated by keeping country‑years where fewer than 50% of months have zero pageviews.

**Interaction:** Hover to explore country‑year points, zoom or pan the chart, and use the toolbar to export the figure.
The plot highlights the mean per region (triangle marker) and labels notable country‑years for context.
""")
st.markdown("---")

# --- Load merged dataset ---
data_path = os.path.join("data", "st01_data.csv")
if not os.path.exists(data_path):
    st.error(f"Required file not found: {data_path}")
    st.stop()
merged_df = pd.read_csv(data_path)

# --- Reconstruct logical tables from merged file ---
if "country_year" not in merged_df.columns and {"country", "year"}.issubset(merged_df.columns):
    merged_df["country_year"] = merged_df["country"].astype(str) + " " + merged_df["year"].astype("Int64").astype(str)

pageviews_year_percapita_continent = merged_df[["region", "country_year", "total_pageviews", "population", "country"]].drop_duplicates().copy()
daily_cc_interest_total_per_country = merged_df[["country", "date", "total_cc_pageview_counts"]].copy()

# Clean types
pageviews_year_percapita_continent["total_pageviews"] = pd.to_numeric(pageviews_year_percapita_continent["total_pageviews"], errors="coerce")
pageviews_year_percapita_continent["population"] = pd.to_numeric(pageviews_year_percapita_continent["population"], errors="coerce")
daily_cc_interest_total_per_country["date"] = pd.to_datetime(daily_cc_interest_total_per_country["date"], errors="coerce")
daily_cc_interest_total_per_country["total_cc_pageview_counts"] = pd.to_numeric(daily_cc_interest_total_per_country["total_cc_pageview_counts"], errors="coerce")

pageviews_year_percapita_continent = pageviews_year_percapita_continent.dropna(subset=["region", "country_year", "total_pageviews", "population", "country"])
daily_cc_interest_total_per_country = daily_cc_interest_total_per_country.dropna(subset=["country", "date", "total_cc_pageview_counts"])

# --- Replicate R logic ---
# Remove Malaysia rows
pageviews_year_percapita_continent = pageviews_year_percapita_continent[
    ~pageviews_year_percapita_continent["country_year"].str.contains("Malaysia", na=False)
].copy()

# Derive country from country_year for safety
pageviews_year_percapita_continent["country"] = pageviews_year_percapita_continent["country_year"].apply(
    lambda x: re.sub(r"\s*\d{4}$", "", x)
)

# Monthly aggregation and zero-month share
daily_cc_interest_total_per_country["month"] = daily_cc_interest_total_per_country["date"].dt.to_period("M")
monthly_pageviews_df = (
    daily_cc_interest_total_per_country.groupby(["country", "month"])["total_cc_pageview_counts"]
    .sum()
    .reset_index()
)

all_countries = monthly_pageviews_df["country"].drop_duplicates()
all_months = monthly_pageviews_df["month"].drop_duplicates()
full_idx = pd.MultiIndex.from_product([all_countries, all_months], names=["country", "month"])
full_months_df = pd.DataFrame(index=full_idx).reset_index()
full_months_df = pd.merge(full_months_df, monthly_pageviews_df, on=["country", "month"], how="left")
full_months_df.rename(columns={"total_cc_pageview_counts": "pageview_month"}, inplace=True)
full_months_df["pageview_month"] = full_months_df["pageview_month"].fillna(0)
full_months_df["zero_month"] = np.where(full_months_df["pageview_month"] == 0, "zero", "non-zero")

month_counts = full_months_df.groupby(["country", "zero_month"]).size().reset_index(name="count_month_types")
zero_month_share = month_counts[month_counts["zero_month"] == "zero"].copy()
total_months_count = full_months_df.groupby("country").size().reset_index(name="total_months")
zero_month_share = zero_month_share.merge(total_months_count, on="country", how="left")
zero_month_share["share_0_month"] = zero_month_share["count_month_types"] / zero_month_share["total_months"] * 100

countries_50_month = zero_month_share[
    (zero_month_share["share_0_month"] < 50) & (zero_month_share["country"] != "Malaysia")
]["country"].tolist()

# Per-capita and log transform (natural log with tiny epsilon to avoid -inf)
safe_den = pageviews_year_percapita_continent["population"].astype(float)
safe_den = safe_den.where(safe_den > 0, np.nan)
pageviews_year_percapita_continent["pageviews_per_capita"] = (
    pageviews_year_percapita_continent["total_pageviews"].astype(float) / safe_den
)
epsilon = 1e-12
pageviews_year_percapita_continent["log_pageviews"] = np.log(
    np.clip(pageviews_year_percapita_continent["pageviews_per_capita"].values, epsilon, None)
)

# Labels (top/bottom 3 per region) among countries with sufficient months
filtered_for_labels = pageviews_year_percapita_continent[
    pageviews_year_percapita_continent["country"].isin(countries_50_month)
].copy()

top_labels = (
    filtered_for_labels.sort_values("log_pageviews", ascending=False)
    .groupby("region", group_keys=False)
    .head(3)["country_year"]
    .tolist()
)
bottom_labels = (
    filtered_for_labels.sort_values("log_pageviews", ascending=True)
    .groupby("region", group_keys=False)
    .head(3)["country_year"]
    .tolist()
)
label_bee_all = top_labels + bottom_labels

# Regions and colors
regions = ["Europe", "Americas", "Oceania", "Asia", "Africa"]
region_color_map = {
    "Europe": "#000000",
    "Americas": "#CC79A7",
    "Oceania": "#009E73",
    "Asia": "#56B4E9",
    "Africa": "#E69F00",
}

# --- Plotting (interactive with Plotly) ---
filtered_df = pageviews_year_percapita_continent[
    pageviews_year_percapita_continent["country"].isin(countries_50_month)
].copy()
filtered_df["region"] = pd.Categorical(filtered_df["region"], categories=regions, ordered=True)

fig = px.strip(
    filtered_df,
    x="log_pageviews",
    y="region",
    color="region",
    color_discrete_map=region_color_map,
    category_orders={"region": regions},
    hover_data={
        "country_year": True,
        "region": True,
        "pageviews_per_capita": True,
        "log_pageviews": ":.3f",
    },
)

# Add mean markers per region
for region_name in regions:
    subset = filtered_df[filtered_df["region"] == region_name]
    if subset.empty:
        continue
    mean_value = subset["log_pageviews"].mean()
    fig.add_trace(
        go.Scatter(
            x=[mean_value],
            y=[region_name],
            mode="markers",
            marker_symbol="triangle-up",
            marker_size=16,
            marker_color=region_color_map.get(region_name, "black"),
            marker_line_color="black",
            marker_line_width=2,
            opacity=1.0,
            showlegend=False,
            hoverinfo="skip",
        )
    )

fig.update_layout(
    showlegend=False,
    height=500,
    xaxis_title="Pageviews per capita (log)\n(n=258)",
    yaxis_title="",
    xaxis=dict(range=[-8, -2], tickvals=[-8, -6, -4, -2], dtick=1, showgrid=True, gridcolor="rgba(0,0,0,0.2)"),
    yaxis=dict(categoryorder="array", categoryarray=regions, autorange="reversed", showgrid=True, gridcolor="rgba(0,0,0,0.2)"),
    margin=dict(l=20, r=20, t=40, b=60)
)

st.plotly_chart(fig, use_container_width=True)
