import pandas as pd
import scipy.stats as stats
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("Readable_data .csv")  # Adjust filename if needed

# Pivot to wide format
df_wide = df.pivot(index="Subject no.", columns="Section no.", values="Score")
df_wide.columns = ["Section_1", "Section_2", "Section_3"]

# Shapiro-Wilk test for normality
print("Shapiro-Wilk Test Results:")
for col in df_wide.columns:
    stat, p = stats.shapiro(df_wide[col])
    print(f"{col}: W={stat:.4f}, p={p:.4f} ({'Not normal' if p < 0.05 else 'Normal'})")

# Histograms & Q-Q plots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, col in enumerate(df_wide.columns):
    sns.histplot(df_wide[col], kde=True, ax=axes[0, i], bins=10, color="skyblue")
    axes[0, i].set_title(f"Histogram of {col}")
    stats.probplot(df_wide[col], dist="norm", plot=axes[1, i])
    axes[1, i].set_title(f"Q-Q Plot of {col}")
plt.tight_layout()
plt.show()


import pandas as pd
import pingouin as pg

# Load data
df = pd.read_csv("Readable_data .csv")

# Ensure columns are named correctly
df.columns = df.columns.str.strip()  # Remove any accidental whitespace

# Ensure that 'Score' is numeric
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

# Run Mauchly’s test for sphericity
mauchly = pg.sphericity(data=df, dv="Score", subject="Subject no.", within="Section no.")

# Print results
print(mauchly)

# Repeated Measures ANOVA
anova = pg.rm_anova(dv="Score", within="Section no.", subject="Subject no.", data=df, detailed=True)
print("\nRepeated Measures ANOVA:")
print(anova)

# Post-hoc tests (Paired t-tests with Bonferroni correction)
pairwise = pg.pairwise_ttests(dv="Score", within="Section no.", subject="Subject no.", data=df, parametric=True, padjust="bonferroni")
print("\nPairwise Comparisons:")
print(pairwise)
