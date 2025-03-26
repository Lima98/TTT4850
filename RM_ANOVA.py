import pandas as pd
import pingouin as pg

# Load dataset
file_path = "Data/Readable_data .csv"
df = pd.read_csv(file_path, delimiter=",")

# Reshape from long format to wide format
df_wide = df.pivot(index="Subject no.", columns="Type", values="Score").reset_index()

# Convert wide format back to long format for RM-ANOVA
df_long = df_wide.melt(id_vars=["Subject no."], var_name="Condition", value_name="Score")

# Run Repeated Measures ANOVA
anova_results = pg.rm_anova(data=df_long, dv="Score", within="Condition", subject="Subject no.")

# Print results
print(anova_results)

# Use the new function name (pairwise_tests instead of pairwise_ttests)
posthoc_results = pg.pairwise_tests(data=df_long, dv="Score", within="Condition", subject="Subject no.", padjust="bonferroni")

# Print post-hoc results
print(posthoc_results)

