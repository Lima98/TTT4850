import pingouin as pg

import pandas as pd
import pingouin as pg

# Load the dataset
df = pd.read_csv("Data/Readable_data .csv", sep=",")  # Ensure ',' is used after replacing ';'

# Convert to wide format (assuming Subject no. identifies subjects)
df_wide = df.pivot(index="Subject no.", columns="Type", values="Score")

# Rename columns for easier access
df_wide.columns = ["K", "P", "S"]


# K vs. P
wilcoxon_KP = pg.wilcoxon(df_wide["K"], df_wide["P"])
print("Wilcoxon K vs. P:\n", wilcoxon_KP)

# K vs. S
wilcoxon_KS = pg.wilcoxon(df_wide["K"], df_wide["S"])
print("Wilcoxon K vs. S:\n", wilcoxon_KS)

# P vs. S
wilcoxon_PS = pg.wilcoxon(df_wide["P"], df_wide["S"])
print("Wilcoxon P vs. S:\n", wilcoxon_PS)

from scipy.stats import mannwhitneyu

# K vs. P
U_KP, p_KP = mannwhitneyu(df_wide["K"], df_wide["P"], alternative='two-sided')
print(f"Mann-Whitney K vs. P: U={U_KP}, p={p_KP}")

# K vs. S
U_KS, p_KS = mannwhitneyu(df_wide["K"], df_wide["S"], alternative='two-sided')
print(f"Mann-Whitney K vs. S: U={U_KS}, p={p_KS}")

# P vs. S
U_PS, p_PS = mannwhitneyu(df_wide["P"], df_wide["S"], alternative='two-sided')
print(f"Mann-Whitney P vs. S: U={U_PS}, p={p_PS}")
