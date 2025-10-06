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
# --------------------------------------------
# Additional Visual Insights (Contribution by Bekhzod Nematjanov)
# --------------------------------------------

# 1️⃣ Pairplot of key numeric variables to explore relationships
sns.pairplot(dataset2, vars=[
    'bat_landing_number',
    'rat_arrival_number',
    'food_availability',
    'rat_minutes'
], hue='season', palette='Set2')
plt.suptitle('Pairwise Relationships Between Bat, Rat, and Food Variables', y=1.02)
plt.show()

# 2️⃣ Scatter plot to visualize relationship between rat arrivals and food availability
plt.figure(figsize=(10,6))
sns.scatterplot(
    x='rat_arrival_number',
    y='food_availability',
    hue='season',
    data=dataset2,
    palette='coolwarm',
    s=60,
    alpha=0.8
)
plt.title('Relationship Between Rat Arrivals and Food Availability by Season')
plt.xlabel('Number of Rat Arrivals')
plt.ylabel('Food Availability')
plt.show()

# 3️⃣ Correlation of Rat and Bat activity vs Food availability (custom insight)
activity_corr = dataset2[['rat_arrival_number', 'bat_landing_number', 'food_availability']].corr()
print("\nCorrelation between Rat/Bat Activity and Food Availability:")
print(activity_corr)

# 4️⃣ Save additional plots 
plt.figure(figsize=(8,5))
sns.barplot(
    x='season',
    y='rat_minutes',
    data=dataset2,
    palette='pastel',
    estimator=np.mean
)
plt.title('Average Rat Minutes Spent on Platform by Season')
plt.xlabel('Season')
plt.ylabel('Mean Rat Minutes')
plt.show()
# --------------------------------------------
# Additional Visual Insights (Contribution by Bekhzod)
# --------------------------------------------

# 1️⃣ Pairplot of key numeric variables to explore relationships
sns.pairplot(dataset2, vars=[
    'bat_landing_number',
    'rat_arrival_number',
    'food_availability',
    'rat_minutes'
], hue='season', palette='Set2')
plt.suptitle('Pairwise Relationships Between Bat, Rat, and Food Variables', y=1.02)
plt.show()

# 2️⃣ Scatter plot to visualize relationship between rat arrivals and food availability
plt.figure(figsize=(10,6))
sns.scatterplot(
    x='rat_arrival_number',
    y='food_availability',
    hue='season',
    data=dataset2,
    palette='coolwarm',
    s=60,
    alpha=0.8
)
plt.title('Relationship Between Rat Arrivals and Food Availability by Season')
plt.xlabel('Number of Rat Arrivals')
plt.ylabel('Food Availability')
plt.show()

# 3️⃣ Correlation of Rat and Bat activity vs Food availability (custom insight)
activity_corr = dataset2[['rat_arrival_number', 'bat_landing_number', 'food_availability']].corr()
print("\nCorrelation between Rat/Bat Activity and Food Availability:")
print(activity_corr)

# 4️⃣ Save additional plots for the report if needed
plt.figure(figsize=(8,5))
sns.barplot(
    x='season',
    y='rat_minutes',
    data=dataset2,
    palette='pastel',
    estimator=np.mean
)
plt.title('Average Rat Minutes Spent on Platform by Season')
plt.xlabel('Season')
plt.ylabel('Mean Rat Minutes')
plt.show()
