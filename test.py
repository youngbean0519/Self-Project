import pandas as pd

columns = ["ID", "text", "labels"]

train = pd.read_csv("data/raw/train.tsv", sep="\t", header=None, names=columns)
val = pd.read_csv("data/raw/val.tsv", sep="\t", header=None, names=columns)
test = pd.read_csv("data/raw/test.tsv", sep="\t", header=None, names=columns)

print("TRAIN")
print(train.shape)
print(train.columns)
print(train.head())

print("\nVALIDATION")
print(val.shape)

print("\nTEST")
print(test.shape)