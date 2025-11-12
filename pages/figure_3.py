import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import re
import os

# --- Load Data from a single merged CSV ---
# Read the already-merged dataset and plot (no merging logic here)
merged_path = "data/st315_data.csv"
merged_df = pd.read_csv(merged_path)

# Reconstruct the two logical tables used below from the merged dataset
# Ensure country_year exists (derive from country and year if not)
if 'country_year' not in merged_df.columns and {'country', 'year'}.issubset(merged_df.columns):
    merged_df['country_year'] = merged_df['country'].astype(str) + ' ' + merged_df['year'].astype('Int64').astype(str)

pageviews_year_percapita_continent = merged_df[[
    "region", "country_year", "total_pageviews", "population", "country"
]].drop_duplicates().copy()
daily_cc_interest_total_per_country = merged_df[[
    "country", "date", "total_cc_pageview_counts"
]].copy()

# Clean and type-cast for downstream computations
pageviews_year_percapita_continent["total_pageviews"] = pd.to_numeric(
    pageviews_year_percapita_continent["total_pageviews"], errors="coerce"
)
pageviews_year_percapita_continent["population"] = pd.to_numeric(
    pageviews_year_percapita_continent["population"], errors="coerce"
)
daily_cc_interest_total_per_country["date"] = pd.to_datetime(
    daily_cc_interest_total_per_country["date"], errors="coerce"
)
daily_cc_interest_total_per_country["total_cc_pageview_counts"] = pd.to_numeric(
    daily_cc_interest_total_per_country["total_cc_pageview_counts"], errors="coerce"
)

# Drop rows that don't belong to each logical table
pageviews_year_percapita_continent = pageviews_year_percapita_continent.dropna(
    subset=["region", "country_year", "total_pageviews", "population", "country"]
)
daily_cc_interest_total_per_country = daily_cc_interest_total_per_country.dropna(
    subset=["country", "date", "total_cc_pageview_counts"]
)
# --- Replicating the R data manipulation ---
# Remove Malaysia as per the R code
pageviews_year_percapita_continent = pageviews_year_percapita_continent[
    ~pageviews_year_percapita_continent['country_year'].str.contains('Malaysia', na=False)
].copy()

# The R code mutates a 'country' column by removing the year. We replicate this.
pageviews_year_percapita_continent['country'] = pageviews_year_percapita_continent['country_year'].apply(
    lambda x: re.sub(r'\s*\d{4}$', '', x)
)

# --- Replicating the R code for `countries_50_month` with the new data structure ---
# Convert the 'date' column to datetime objects
daily_cc_interest_total_per_country['date'] = pd.to_datetime(daily_cc_interest_total_per_country['date'])
# Create a unique month identifier
daily_cc_interest_total_per_country['month'] = daily_cc_interest_total_per_country['date'].dt.to_period('M')
print(daily_cc_interest_total_per_country['month'][:10])
# Group by country and month to get monthly pageviews
monthly_pageviews_df = daily_cc_interest_total_per_country.groupby(
    ['country', 'month']
)['total_cc_pageview_counts'].sum().reset_index()

# 1. Get all unique country and month combinations.
all_countries = monthly_pageviews_df['country'].drop_duplicates()
all_months = monthly_pageviews_df['month'].drop_duplicates()
full_country_month_combinations = pd.MultiIndex.from_product([all_countries, all_months], names=['country', 'month'])
full_months_df = pd.DataFrame(index=full_country_month_combinations).reset_index()
print(full_months_df[:10])
print("*")
print(monthly_pageviews_df[:10])
print("*")
# 2. Left join the original data to get missing months with 0 pageviews
full_months_df = pd.merge(full_months_df, monthly_pageviews_df, on=['country', 'month'], how='left')
# 3. Fill missing pageviews with 0 and check for zero months
full_months_df.rename(columns={'total_cc_pageview_counts': 'pageview_month'}, inplace=True)
print(full_months_df[:10])

full_months_df['pageview_month'].fillna(0, inplace=True)
full_months_df['zero_month'] = np.where(full_months_df['pageview_month'] == 0, 'zero', 'non-zero')

# 4. Count the number of unique months per country and zero_month category
month_counts = full_months_df.groupby(['country', 'zero_month']).size().reset_index(name='count_month_types')

# 5. Filter for 'zero' months and calculate the share
zero_month_share = month_counts[month_counts['zero_month'] == 'zero'].copy()
total_months_count = full_months_df.groupby('country').size()
zero_month_share = pd.merge(zero_month_share, total_months_count.reset_index(name='total_months'), on='country', how='left')
zero_month_share['share_0_month'] = zero_month_share['count_month_types'] / zero_month_share['total_months'] * 100

# 6. Filter for countries with less than 50% zero months and not 'Malaysia'
countries_50_month = zero_month_share[
    (zero_month_share['share_0_month'] < 50) & (zero_month_share['country'] != 'Malaysia')
]['country'].tolist()

# Calculate pageviews per capita and log-transform for both plots
# Use a tiny epsilon to avoid -inf for exact zeros without compressing the scale
safe_den = pageviews_year_percapita_continent['population'].astype(float)
safe_den = safe_den.where(safe_den > 0, np.nan)
pageviews_year_percapita_continent['pageviews_per_capita'] = (
    pageviews_year_percapita_continent['total_pageviews'].astype(float) / safe_den
)
epsilon = 1e-12
pageviews_year_percapita_continent['log_pageviews'] = np.log(
    np.clip(pageviews_year_per_capita := pageviews_year_percapita_continent['pageviews_per_capita'].values, epsilon, None)
)

# --- Dynamically generating labels for the beeswarm plot (label_bee_all) ---
# Filter to only include countries in `countries_50_month`
filtered_for_labels = pageviews_year_percapita_continent[
    pageviews_year_percapita_continent['country'].isin(countries_50_month)
].copy()

# Find the top 3 country-year combinations per region
top_labels = filtered_for_labels.groupby('region').apply(
    lambda x: x.nlargest(3, 'log_pageviews')['country_year']
).explode().tolist()

# Find the bottom 3 country-year combinations per region
bottom_labels = filtered_for_labels.groupby('region').apply(
    lambda x: x.nsmallest(3, 'log_pageviews')['country_year']
).explode().tolist()

label_bee_all = top_labels + bottom_labels

# Define the color palette used in the R code
palette_okabe_country = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2']
# Fixed region -> color mapping to match the reference visual
region_color_map = {
    'Europe': '#000000',    # black
    'Americas': '#CC79A7',  # magenta
    'Oceania': '#009E73',   # green
    'Asia': '#56B4E9',      # light blue
    'Africa': '#E69F00',    # orange
}

# --- Start of Plotting ---
plt.style.use('seaborn-v0_8-ticks')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# --- Plot: Beeswarm (Recreating country_year_pc_beeswarm) ---
# Reorder regions as a categorical type for desired ordering
regions = ['Europe', 'Americas', 'Oceania', 'Asia', 'Africa']
pageviews_year_percapita_continent['region'] = pd.Categorical(
    pageviews_year_percapita_continent['region'], categories=regions, ordered=True
)
# Single-axis figure focused on plot b)
fig, ax2 = plt.subplots(figsize=(18, 7))

# Filter, mutate, and create labels
filtered_df = pageviews_year_percapita_continent[
    pageviews_year_percapita_continent['country'].isin(countries_50_month)
].copy()

# Fix: Explicitly re-apply the categorical type to the filtered DataFrame
filtered_df['region'] = pd.Categorical(filtered_df['region'], categories=regions, ordered=True)

filtered_df['label'] = filtered_df.apply(
    lambda row: row['country_year'] if row['country_year'] in label_bee_all else '',
    axis=1
)

# Recreate the beeswarm plot using seaborn's swarmplot
sns.stripplot(
    data=filtered_df,
    x='log_pageviews',
    y='region',
    hue='region',
    palette={r: region_color_map.get(r, '#333333') for r in regions},
    ax=ax2,
    size=6,
)

# Add grid and remove borders
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Recreate stat_summary for the mean points (triangles)
for region_name in filtered_df['region'].unique():
    subset = filtered_df[filtered_df['region'] == region_name]
    mean_value = subset['log_pageviews'].mean()
    y_pos = filtered_df['region'].cat.categories.get_loc(region_name)
    ax2.plot(mean_value, y_pos, marker='^', markersize=12, color=region_color_map.get(region_name, 'black'))

# Create a new DataFrame for just the labels to make positioning easier
label_df = filtered_df[filtered_df['label'] != ''].copy()

# Sort the label_df by region and then by log_pageviews to get a consistent order
label_df['label_rank'] = label_df.groupby('region')['log_pageviews'].rank(method='first', ascending=True)

# Assign unique vertical offsets based on rank to prevent overlap
offset_step = 0.18
offset_map = {1: -2 * offset_step, 2: -offset_step, 3: 0, 4: offset_step, 5: 2 * offset_step, 6: 3 * offset_step}
label_df['offset_y'] = label_df['label_rank'].map(offset_map)

# Now, iterate over this simplified dataframe to place the labels
for row in label_df.itertuples():
    y_pos = filtered_df['region'].cat.categories.get_loc(row.region)
    ax2.text(
        row.log_pageviews + 0.1,  # x-offset
        y_pos + row.offset_y,    # y-offset based on rank
        row.label,
        ha='left',
        va='center',
        fontsize=12,
        color=region_color_map.get(row.region, 'black'),
        clip_on=False,  # Set clip_on to False to prevent clipping
    )

# Set x-axis to match reference figure and add gridlines
ax2.set_xlim(-8, -2)
ax2.set_xticks([-8, -6, -4, -2])
ax2.set_xticks(np.arange(-8, -1, 1), minor=True)
ax2.grid(True, which='major', linestyle='--', alpha=0.6)
ax2.grid(True, which='minor', linestyle='--', alpha=0.3)

# Labs and theme_minimal equivalent
ax2.set_xlabel('Pageviews per capita (log)\n(n=258)')
ax2.set_ylabel('')
ax2.set_title('b)', loc='left', pad=10)
#ax2.get_legend().remove()


# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
