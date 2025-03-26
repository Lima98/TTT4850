import pandas as pd
import pingouin as pg

# Load your dataset (replace 'your_file.csv' with the actual filename)
df = pd.read_csv("Data/Readable_data .csv")

# Convert data to wide format (each subject as a row, each condition as a column)
df_wide = df.pivot(index="Subject no.", columns="Type", values="Score")

# Run Mauchly’s test for sphericity
mauchly_test = pg.sphericity(df_wide)

# Print the results
print(mauchly_test)
