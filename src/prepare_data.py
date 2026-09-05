import pandas as pd

# Load both files
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Add label column: 1 = real, 0 = fake
fake_df["label"] = 0
true_df["label"] = 1

# Combine both into one dataset
combined_df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle the rows so real/fake aren't grouped together
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Peek at the result
print(combined_df.head())
print("\nTotal articles:", len(combined_df))
print("\nLabel counts:")
print(combined_df["label"].value_counts())

# Save this combined dataset so we don't have to redo this step each time
combined_df.to_csv("data/combined_dataset.csv", index=False)
print("\nSaved combined_dataset.csv to data folder")