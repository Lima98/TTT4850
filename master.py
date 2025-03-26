import pandas as pd
import scipy.stats as stats
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Load your dataset (replace 'your_file.csv' with actual filename)
df = pd.read_csv("Data/Readable_data .csv")

### OVERALL STATISTICS ###

# Compute overall statistics
mean_all = df["Score"].mean()
std_all = df["Score"].std()

# Compute statistics per condition
mean_K = df[df["Type"] == "K"]["Score"].mean()
std_K = df[df["Type"] == "K"]["Score"].std()

mean_P = df[df["Type"] == "P"]["Score"].mean()
std_P = df[df["Type"] == "P"]["Score"].std()

mean_S = df[df["Type"] == "S"]["Score"].mean()
std_S = df[df["Type"] == "S"]["Score"].std()

# Print results
print("=== Overall Statistics ===")
print(f"Overall:    Mean = {mean_all:.2f}, Std Dev = {std_all:.2f}")
print(f"K:          Mean = {mean_K:.2f}, Std Dev = {std_K:.2f}")
print(f"P:          Mean = {mean_P:.2f}, Std Dev = {std_P:.2f}")
print(f"S:          Mean = {mean_S:.2f}, Std Dev = {std_S:.2f}")
print("==========================\n")


### NORMALITY CHECK ###

# Convert from long format to wide format
df_wide = df.pivot(index="Subject no.", columns="Type", values="Score")

# Rename columns for clarity (if necessary)
df_wide.columns = ["K", "P", "S"]

# Shapiro-Wilk test (Recommended for small samples, n < 50)
shapiro_K = stats.shapiro(df_wide["K"])
shapiro_P = stats.shapiro(df_wide["P"])
shapiro_S = stats.shapiro(df_wide["S"])

print("\n=== Normality Check ===")
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
print("=======================\n")

# Create subplots for Histograms & Q-Q plots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# List of columns
conditions = ["K", "P", "S"]

for i, condition in enumerate(conditions):
    # Histogram
    sns.histplot(df_wide[condition], kde=True, ax=axes[0, i], bins=10, color="skyblue")
    axes[0, i].set_title(f"Histogram of {condition}")

    # Q-Q Plot
    stats.probplot(df_wide[condition], dist="norm", plot=axes[1, i])
    axes[1, i].set_title(f"Q-Q Plot of {condition}")

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



### RM-ANOVA & POST-HOC TESTS ###

## Sphrericity Test for RM-ANOVA ##
mauchly_test = pg.sphericity(df_wide)

# Extract values for readability
W = mauchly_test.W
chi2 = mauchly_test.chi2
dof = int(mauchly_test.dof)
p_value = mauchly_test.pval

# Print results
print("\n=== Mauchly’s Test for Sphericity ===")
print(f" Mauchly’s W:           {W:.4f}")
print(f" Chi-square:            {chi2:.4f}")
print(f" Degrees of freedom:    {dof}")
print(f" p-value:               {p_value:.4f}")

# Interpretation based on p-value
if p_value < 0.05:
    print(" Sphericity is violated (p < 0.05).")
else:
    print(" Sphericity is not violated (p ≥ 0.05).")
print("======================================\n")

## Repeated Measures RM-ANOVA ##
df = pd.read_csv("Data/Readable_data .csv", delimiter=",")

# Reshape from long format to wide format
df_wide = df.pivot(index="Subject no.", columns="Type", values="Score").reset_index()

# Convert wide format back to long format for RM-ANOVA
df_long = df_wide.melt(id_vars=["Subject no."], var_name="Condition", value_name="Score")

# Run Repeated Measures ANOVA
anova_results = pg.rm_anova(data=df_long, dv="Score", within="Condition", subject="Subject no.")

# Print results
print("\n=== Repeated Measures ANOVA ===")
print(anova_results)
print("===============================\n")

# Use the new function name (pairwise_tests instead of pairwise_ttests)
posthoc_results = pg.pairwise_tests(data=df_long, dv="Score", within="Condition", subject="Subject no.", padjust="bonferroni")

# Print post-hoc results
print("\n=== Post-Hoc Tests to Perform Pairwise Comparison ===")
print(posthoc_results)
print("=====================================================\n")
