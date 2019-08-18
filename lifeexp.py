import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt

data = pd.read_csv("Life Expectancy Data.csv")

pd.set_option('display.max_columns', 10)
#
# print(data.info())
# print(data.isna().sum())
data = data.dropna()
# print(data.isna().sum())
#
# print(data.info())
# print(data.dtypes)
print(data.corr())

plt.hist(data.Life_expectancy)
plt.show()

x_vals =["BMI", "Polio", "Diphtheria", "HIVAIDS", "GDP"]
# print(data[x_vals])

linearModel = LinearRegression()
ensembleModel = RandomForestRegressor()

cValLinear = cross_val_score(linearModel, data[x_vals], data["Life_expectancy"], cv=10)
cValEnsemble = cross_val_score(ensembleModel, data[x_vals], data["Life_expectancy"], cv=10)


print(cValLinear)
print(cValEnsemble)
