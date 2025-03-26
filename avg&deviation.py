import pandas as pd

# Load your dataset (replace 'your_file.csv' with actual filename)
df = pd.read_csv("Readable_data .csv")

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
print(f"Overall: Mean = {mean_all:.2f}, Std Dev = {std_all:.2f}")
print(f"K: Mean = {mean_K:.2f}, Std Dev = {std_K:.2f}")
print(f"P: Mean = {mean_P:.2f}, Std Dev = {std_P:.2f}")
print(f"S: Mean = {mean_S:.2f}, Std Dev = {std_S:.2f}")
