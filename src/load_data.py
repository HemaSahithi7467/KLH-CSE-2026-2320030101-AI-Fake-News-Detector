import pandas as pd

# Load both files
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Peek at what's inside
print("FAKE NEWS SAMPLE:")
print(fake_df.head())

print("\nREAL NEWS SAMPLE:")
print(true_df.head())

print("\nNumber of fake articles:", len(fake_df))
print("Number of real articles:", len(true_df))