import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load dataset
file_path = "Data/Readable_data .csv"
df = pd.read_csv(file_path, delimiter=",")

# Descriptive statistics
summary = df.groupby("Type")["Score"].describe()
print("Summary Statistics:\n", summary)

# Boxplot of scores by noise type
plt.figure(figsize=(8, 6))
sns.boxplot(x="Type", y="Score", data=df, palette="Set2")
plt.title("Score Distribution by Noise Type")
plt.xlabel("Noise Type")
plt.ylabel("Score")
plt.show()

# ANOVA test
anova_result = stats.f_oneway(
    df[df["Type"] == "S"]["Score"],
    df[df["Type"] == "P"]["Score"],
    df[df["Type"] == "K"]["Score"]
)
print("ANOVA Test Result:", anova_result)

# Post-hoc test if ANOVA is significant
if anova_result.pvalue < 0.05:
    tukey = pairwise_tukeyhsd(df["Score"], df["Type"], alpha=0.05)
    print("\nPost-hoc Tukey Test:\n", tukey)
else:
    print("\nNo significant difference found between groups.")
