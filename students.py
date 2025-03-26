import scipy.stats as stats

import pandas as pd

# Load your dataset
df = pd.read_csv("Data/Readable_data .csv")  # Ensure this matches your file location

# Convert to wide format (each subject has one row)
df_wide = df.pivot(index="Subject no.", columns="Type", values="Score")

# Rename columns for clarity (if necessary)
df_wide.columns = ["K", "P", "S"]

# Print to check the transformation
print(df_wide.head())


# Perform paired t-tests
t_test_KP = stats.ttest_rel(df_wide["K"], df_wide["P"])
t_test_KS = stats.ttest_rel(df_wide["K"], df_wide["S"])
t_test_PS = stats.ttest_rel(df_wide["P"], df_wide["S"])

# Print results
print("Paired t-tests:")
print(f"K vs. P: t={t_test_KP.statistic:.4f}, p={t_test_KP.pvalue:.4f}")
print(f"K vs. S: t={t_test_KS.statistic:.4f}, p={t_test_KS.pvalue:.4f}")
print(f"P vs. S: t={t_test_PS.statistic:.4f}, p={t_test_PS.pvalue:.4f}")

from statsmodels.stats.multitest import multipletests

p_values = [t_test_KP.pvalue, t_test_KS.pvalue, t_test_PS.pvalue]
corrected_p = multipletests(p_values, method="bonferroni")[1]

print("Bonferroni corrected p-values:", corrected_p)
