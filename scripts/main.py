import pandas as pd
import numpy as np
from manual import mean, variance, std_dev
from kNearest import predict

df = pd.read_csv("data\Mock-People.csv")
df['label'] = (df['age'] > 50).astype(int)

print(mean(df["age"]))
print(variance(df["age"]))
print(std_dev(df["age"]))

test = np.array([25, 110000, 75000, 45, 7]) 

print(predict(df, test, k=5))

# Output: 1 even though the test data is for a younger person, the model predicts that they are older than 50. This is likely due to the fact that the test data has a high income correlated with age.

