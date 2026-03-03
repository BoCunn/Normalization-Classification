import pandas as pd
import numpy as np
from manual import mean, variance, std_dev
from kNearest import predict, knn
from normalize import Z_Score_normalize, min_max_normalize


# here the data is labeled with 1 if the person is older than 50 and 0 if they are younger. The test data is for a younger person, but it has a high income which may lead the model to predict that they are older than 50.
cols = ['age', 'workclass', 'fnlwgt', 'education', 'ednum', 'maritalstat', 'occ', 'relation', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv("data/adult.csv", names=cols, skipinitialspace=True)
df = df[[
    "age",
    "ednum", "sex",
    "hours-per-week",
    "income"
]]
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['ednum'] = pd.to_numeric(df['ednum'], errors='coerce')
df['hours-per-week'] = pd.to_numeric(df['hours-per-week'], errors='coerce')
df['sex'] = (df['sex'] == 'Male').astype(int)
df['income'] = (df['income'] =='>50K').astype(int)

print(mean(df["age"]))
print(variance(df["age"]))
print(std_dev(df["age"]))




test_data = np.array([39, 25, 40, 1])
print(predict(df, test_data, 5))
# Output: this prediction returns 0, meaning that the model predicts that this person (39 yo, 25 years of education, 40 hours a week, male) makes less than 50K a year

test_data = np.array([59, 25, 40, 1])
print(predict(df, test_data, 5))
# Output: this prediction returns 1, meaning that the model predicts that this person (59 yo, 25 years of education, 40 hours a week, male) makes more than 50K a year

#this likely means there is some correlation between age and income in the dataset, which is not surprising. However, it also means that the model may not be very accurate for younger people with high incomes, as it may be biased towards predicting that they are older than 50.

dfZnormalized = df.copy()
dfZnormalized["age"] = Z_Score_normalize(dfZnormalized["age"])
dfZnormalized["ednum"] = Z_Score_normalize(dfZnormalized["ednum"])
dfZnormalized["hours-per-week"] = Z_Score_normalize(dfZnormalized["hours-per-week"])
dfZnormalized["sex"] = Z_Score_normalize(dfZnormalized["sex"])

test_data = np.array([39, 25, 40, 1])
print(predict(dfZnormalized, test_data, 5))
# Output: this prediction returns 0, meaning that the model predicts that this person (39 yo, 25 years of education, 40 hours a week, male) makes less than 50K a year, with data that is z-score normalized
test_data = np.array([59, 25, 40, 1])
print(predict(dfZnormalized, test_data, 5))
# Output: this prediction returns 0, meaning that the model predicts that this person (59 yo, 25 years of education, 40 hours a week, male) makes less than 50K a year, with data that is z-score normalized. This is likely because the z-score normalization has reduced the influence of age on the model's predictions, making it less likely to predict that older people make more than 50K a year.

dfMinMaxnormalized = df.copy()
dfMinMaxnormalized["age"] = min_max_normalize(dfMinMaxnormalized["age"])
dfMinMaxnormalized["ednum"] = min_max_normalize(dfMinMaxnormalized["ednum"])
dfMinMaxnormalized["hours-per-week"] = min_max_normalize(dfMinMaxnormalized["hours-per-week"])
dfMinMaxnormalized["sex"] = min_max_normalize(dfMinMaxnormalized["sex"])

test_data = np.array([39, 25, 40, 1])
print(predict(dfMinMaxnormalized, test_data, 5))    
# Output: this prediction returns 0, meaning that the model predicts that this person (39 yo, 25 years of education, 40 hours a week, male) makes less than 50K a year, with data that is min-max normalized.
test_data = np.array([59, 25, 40, 1])
print(predict(dfMinMaxnormalized, test_data, 5))
# Output: this prediction returns 1, meaning that this person (59 yo, 25 years of education, 40 hours, male) makes more than 50K a year, with data that is min-max normalized. This is likely because the min-max normalization has not reduced the influence of age on the model's predictions as much as the z-score normalization, making it more likely to predict that older people make more than 50K a year.

print("Data set:                                           Default,     Z-Score Normalized,     Min-Max Normalized")
print("Test data 1 (younger person with high income):         0                  0                         0")
print("Test data 2 (older person with high income):           1                  0                         1")


print("This shows that the kNN model is sensitive to the scale of the data, and that normalizing the data can have a significant impact on the model's predictions. In this case, the z-score normalization has reduced the influence of age on the model's predictions, while the min-max normalization has not reduced it as much.")
