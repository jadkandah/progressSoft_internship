from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris(as_frame=True)

# Create DataFrame
df = iris.frame

# Rename target column
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df = df.drop(columns="target")

# Save to CSV
df.to_csv("iris.csv", index=False)

print(df.head())