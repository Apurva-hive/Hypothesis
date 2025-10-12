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
# ================================
# Figures 8–10: Linear Regression (minimal, additive)
# Response: bat_landing_number  |  Predictors: dataset2 features
# ================================
import statsmodels.api as sm

# --- Prep predictors & response (safe coercion) ---
features = ['rat_arrival_number', 'food_availability', 'rat_minutes',
            'rat_encounter_frequency', 'food_rat_interaction', 'hours_after_sunset']
use_cols = [c for c in features if c in dataset2.columns]

df_lr = dataset2.copy()
# guard against divide-by-zero earlier
df_lr['rat_encounter_frequency'] = df_lr['rat_encounter_frequency'].replace([np.inf, -np.inf], np.nan)

# Drop rows with missing in y or X
df_lr = df_lr.dropna(subset=['bat_landing_number'] + use_cols).copy()

# If too few rows, bail gracefully
if df_lr.shape[0] >= 25 and len(use_cols) >= 2:
    # --------------------------
    # Overall model (Investigation A)
    # --------------------------
    X = sm.add_constant(df_lr[use_cols])
    y = df_lr['bat_landing_number']
    model_all = sm.OLS(y, X).fit()

    # === Figure 8: Residuals vs Fitted (overall) ===
    plt.figure(figsize=(8,6))
    sns.residplot(x=model_all.fittedvalues, y=model_all.resid, lowess=True, line_kws={'color':'red'})
    plt.title("Figure 8. Residuals vs Fitted — Overall Linear Regression")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.show()

    # === Figure 9: Coefficient magnitudes (overall) ===
    coef_all = model_all.params.drop(labels=['const'], errors='ignore').sort_values()
    plt.figure(figsize=(9,6))
    coef_all.plot(kind='barh')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.title("Figure 9. Coefficient Estimates — Overall Model")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.show()

    # --------------------------
    # Seasonal models (Investigation B): Winter vs Spring
    # --------------------------
    seasons_to_compare = ['Winter', 'Spring']
    coef_compare = []

    for s in seasons_to_compare:
        sub = df_lr[df_lr['season'] == s]
        # need enough rows to fit
        if sub.shape[0] >= max(15, 2*len(use_cols)):
            Xs = sm.add_constant(sub[use_cols])
            ys = sub['bat_landing_number']
            m = sm.OLS(ys, Xs).fit()
            for v in use_cols:
                coef_compare.append({'Variable': v, 'Season': s, 'Coefficient': m.params.get(v, np.nan)})
        else:
            # still record NaNs to keep alignment
            for v in use_cols:
                coef_compare.append({'Variable': v, 'Season': s, 'Coefficient': np.nan})

    coef_df = pd.DataFrame(coef_compare)
    if not coef_df.empty:
        pivot_coef = coef_df.pivot(index='Variable', columns='Season', values='Coefficient').loc[use_cols]

        # === Figure 10: Seasonal Coefficient Comparison (Winter vs Spring) ===
        pivot_coef.plot(kind='bar', figsize=(10,6))
        plt.title("Figure 10. Seasonal Coefficient Comparison — Winter vs Spring")
        plt.ylabel("Coefficient")
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.tight_layout()
        plt.show()

        # (Optional) print quick R² for context
        try:
            # recompute simple R²s for transparency
            winter_sub = df_lr[df_lr['season']=='Winter']
            spring_sub = df_lr[df_lr['season']=='Spring']
            if winter_sub.shape[0] >= max(15, 2*len(use_cols)):
                R2_w = sm.OLS(winter_sub['bat_landing_number'], sm.add_constant(winter_sub[use_cols])).fit().rsquared
            else:
                R2_w = np.nan
            if spring_sub.shape[0] >= max(15, 2*len(use_cols)):
                R2_s = sm.OLS(spring_sub['bat_landing_number'], sm.add_constant(spring_sub[use_cols])).fit().rsquared
            else:
                R2_s = np.nan
            print(f"\nSeasonal R² — Winter: {R2_w:.3f} | Spring: {R2_s:.3f}")
        except Exception as e:
            print("Note: could not compute seasonal R² summary:", e)
    else:
        print("Note: Not enough seasonal data to plot Figure 10.")

else:
    print("Note: Not enough rows or predictors for LR figures (need ≥25 rows and ≥2 predictors).")
import os, json, numpy as np, matplotlib.pyplot as plt
from scipy.stats import spearmanr

outB = "contrib_B"; os.makedirs(outB, exist_ok=True)

try:
    dfB = dataset2[['bat_landing_number','food_availability']].dropna()
except NameError:
    import pandas as pd
    dataset2 = pd.read_csv('dataset2.csv')
    dfB = dataset2[['bat_landing_number','food_availability']].dropna()

if len(dfB) >= 10:
    r, p = spearmanr(dfB['bat_landing_number'], dfB['food_availability'])
    with open(f"{outB}/spearman_bat_food.json", "w") as f:
        json.dump({"spearman_r": float(r), "p_value": float(p), "n": int(len(dfB))}, f, indent=2)

    plt.figure(figsize=(5,4))
    x = dfB['food_availability'].to_numpy()
    y = dfB['bat_landing_number'].to_numpy()
    plt.scatter(x, y, alpha=0.6)
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    plt.plot(xs, m*xs + b, linestyle="--")
    plt.xlabel("Food availability")
    plt.ylabel("Bat landings")
    plt.title(f"Bat vs Food (Spearman r={r:.2f}, p={p:.3f})")
    plt.tight_layout()
    plt.savefig(f"{outB}/bat_vs_food.png", dpi=200)
    plt.close()

    print(f"[B] Spearman r={r:.2f}, p={p:.3f}, n={len(dfB)}")
else:
    print("[B] Not enough rows for Spearman correlation.")
