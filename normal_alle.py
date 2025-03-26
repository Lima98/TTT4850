import scipy.stats as stats
import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data (adjust the path as needed)
df = pd.read_csv("Data/Readable_data .csv")  # Or use sep="," if needed

# Convert from long format to wide format
df_wide = df.pivot(index="Subject no.", columns="Type", values="Score")

# Rename columns for clarity (if necessary)
df_wide.columns = ["K", "P", "S"]  # Assuming 'K' = Cantina, 'P' = Podcast, 'S' = Silence

# Check the first rows to ensure it's structured correctly
print(df_wide.head())

# Shapiro-Wilk test (Recommended for small samples, n < 50)
shapiro_K = stats.shapiro(df_wide["K"])
shapiro_P = stats.shapiro(df_wide["P"])
shapiro_S = stats.shapiro(df_wide["S"])

print("Shapiro-Wilk Test Results:")
print(f"K: W={shapiro_K.statistic:.4f}, p={shapiro_K.pvalue:.4f}")
print(f"P: W={shapiro_P.statistic:.4f}, p={shapiro_P.pvalue:.4f}")
print(f"S: W={shapiro_S.statistic:.4f}, p={shapiro_S.pvalue:.4f}")

# Interpret results
alpha = 0.05  # Common significance level
for name, test in zip(["K", "P", "S"], [shapiro_K, shapiro_P, shapiro_S]):
    if test.pvalue < alpha:
        print(f"❌ {name} is NOT normally distributed (p = {test.pvalue:.4f})")
    else:
        print(f"✅ {name} is normally distributed (p = {test.pvalue:.4f})")

# Create subplots for Histograms & Q-Q plots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# First column: All Data
# Histogram for all data combined
sns.histplot(pd.concat([df_wide["K"], df_wide["P"], df_wide["S"]]), kde=True, ax=axes[0, 0], bins=10, color="skyblue")
axes[0, 0].set_title("Histogram of All Data")
axes[0, 0].set_xlabel("Score")
axes[0, 0].set_ylabel("Frequency")

# Q-Q plot for all data combined
stats.probplot(pd.concat([df_wide["K"], df_wide["P"], df_wide["S"]]), dist="norm", plot=axes[1, 0])
axes[1, 0].set_title("Q-Q Plot of All Data")

# Next columns: K, P, and S Conditions
conditions = ["K", "P", "S"]

for i, condition in enumerate(conditions):
    # Histogram for individual conditions
    sns.histplot(df_wide[condition], kde=True, ax=axes[0, i+1], bins=10, color="skyblue")
    axes[0, i+1].set_title(f"Histogram of {condition}")
    axes[0, i+1].set_xlabel("Score")
    axes[0, i+1].set_ylabel("Frequency")

    # Q-Q Plot for individual conditions
    stats.probplot(df_wide[condition], dist="norm", plot=axes[1, i+1])
    axes[1, i+1].set_title(f"Q-Q Plot of {condition}")

plt.tight_layout()
plt.show()

# Boxplot & Violin plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Boxplot
sns.boxplot(data=df_wide, ax=axes[0])
axes[0].set_title("Boxplot of Scores")

# Violin plot
sns.violinplot(data=df_wide, ax=axes[1], inner="quartile")
axes[1].set_title("Violin Plot of Scores")

plt.show()
