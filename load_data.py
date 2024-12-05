from datasets import load_dataset
import pandas as pd

# Load the emotion dataset from Hugging Face
data = load_dataset("dair-ai/emotion")

# Convert the train/test splits to pandas dfs
train_df = data["train"].to_pandas()
test_df = data["test"].to_pandas()

# Save the dfs as csv files
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Dataset loaded and saved as train.csv and test.csv")
