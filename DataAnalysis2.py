import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load Dataset 2
dataset2 = pd.read_csv('dataset2.csv')

# Data Cleaning
# Check for duplicates and remove them
dataset2 = dataset2.drop_duplicates()

# Feature Engineering: Map months to seasons
season_map = {
    1: 'Winter', 2: 'Winter', 12: 'Winter',  # December, January, February
    3: 'Spring', 4: 'Spring', 5: 'Spring',    # March, April, May
    6: 'Summer', 7: 'Summer', 8: 'Summer',    # June, July, August
    9: 'Fall', 10: 'Fall', 11: 'Fall'         # September, October, November
}

# Extract month from 'time' column and map to season
dataset2['month'] = pd.to_datetime(dataset2['time']).dt.month
dataset2['season'] = dataset2['month'].map(season_map)

# Feature Engineering: Rat encounter frequency
dataset2['rat_encounter_frequency'] = dataset2['rat_arrival_number'] / dataset2['rat_minutes']

# Interaction term between food availability and rat arrival number
dataset2['food_rat_interaction'] = dataset2['food_availability'] * dataset2['rat_arrival_number']

# Descriptive statistics for numerical columns
desc_stats = dataset2.describe()

# Visualize Rat Arrivals and Bat Landings by Season
plt.figure(figsize=(12, 6))
sns.boxplot(x='season', y='rat_arrival_number', data=dataset2, palette='Set3', legend=False)
plt.title('Distribution of Rat Arrivals by Season')
plt.xlabel('Season')
plt.ylabel('Number of Rat Arrivals')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='season', y='bat_landing_number', data=dataset2, palette='Set3', legend=False)
plt.title('Distribution of Bat Landings by Season')
plt.xlabel('Season')
plt.ylabel('Number of Bat Landings')
plt.show()

# Visualizing Correlations (only numeric columns)
numeric_cols = dataset2.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(12, 8))
sns.heatmap(dataset2[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for Dataset 2')
plt.show()

# Filter out seasons with insufficient data (optional, if necessary)
filtered_data = dataset2.groupby('season').filter(lambda x: len(x) > 5)

# Exclude seasons with zero rat arrivals or very small sample sizes (optional)
filtered_data = filtered_data[filtered_data['rat_arrival_number'] > 0]

# Hypothesis Testing: ANOVA for rat arrivals between season groups
anova_rats = stats.f_oneway(
    filtered_data[filtered_data['season'] == 'Winter']['rat_arrival_number'],
    filtered_data[filtered_data['season'] == 'Spring']['rat_arrival_number'],
)

# ANOVA for bat landings between season groups
anova_bats = stats.f_oneway(
    filtered_data[filtered_data['season'] == 'Winter']['bat_landing_number'],
    filtered_data[filtered_data['season'] == 'Spring']['bat_landing_number'],
)

print(f"ANOVA for Rat Arrivals by Season: F-stat={anova_rats.statistic:.2f}, p-value={anova_rats.pvalue:.2f}")
print(f"ANOVA for Bat Landings by Season: F-stat={anova_bats.statistic:.2f}, p-value={anova_bats.pvalue:.2f}")

# Non-parametric test (Kruskal-Wallis) for rat arrivals between season groups
kruskal_rats = stats.kruskal(
    filtered_data[filtered_data['season'] == 'Winter']['rat_arrival_number'],
    filtered_data[filtered_data['season'] == 'Spring']['rat_arrival_number'],
)

# Non-parametric test for bat landings between season groups
kruskal_bats = stats.kruskal(
    filtered_data[filtered_data['season'] == 'Winter']['bat_landing_number'],
    filtered_data[filtered_data['season'] == 'Spring']['bat_landing_number'],
)

print(f"Kruskal-Wallis for Rat Arrivals by Season: H-stat={kruskal_rats.statistic:.2f}, p-value={kruskal_rats.pvalue:.2f}")
print(f"Kruskal-Wallis for Bat Landings by Season: H-stat={kruskal_bats.statistic:.2f}, p-value={kruskal_bats.pvalue:.2f}")

# Visualize food availability trends across season groups
plt.figure(figsize=(12, 6))
sns.boxplot(x='season', y='food_availability', data=filtered_data, palette='Set3', legend=False)
plt.title('Food Availability by Season')
plt.xlabel('Season')
plt.ylabel('Food Availability')
plt.show()

# Displaying season summary statistics
season_summary = filtered_data.groupby('season').agg({
    'bat_landing_number': ['mean', 'std'],
    'rat_arrival_number': ['mean', 'std'],
    'food_availability': ['mean', 'std'],
    'rat_encounter_frequency': ['mean', 'std'],
    'food_rat_interaction': ['mean', 'std']
})
print(season_summary)
