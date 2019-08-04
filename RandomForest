# Import Library to read csv from Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
%matplotlib inline

# Read CSV 
df_train = pd.read_csv('gdrive/My Drive/Colab Notebooks/House Cost Prediction/train.csv')
df_test = pd.read_csv('gdrive/My Drive/Colab Notebooks/House Cost Prediction/test.csv')
df_train.head()

df_train.shape
# df_train.info()
# df_test.shape
# df_test.info()

# Target Variable
# Some Analysis on the Traget Variable

plt.subplots(figsize=(12,9))
sns.distplot(df_train['SalePrice'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(df_train['SalePrice'])

# plot with the distribution

plt.legend(['Normal Dist. ( mu = {:.2f} and sigma = {:.2f})'.format(mu, sigma)], loc = 'best')
plt.ylabel('Frequency')

# Probablity plot

fig = plt.figure()
stats.probplot(df_train['SalePrice'], plot = plt)
plt.show()

# Here we use log for target variable to make more normal distribution
# we use log function which is in numpy

# df_train['SalePrice'] = np.log1p(df_train['SalePrice'])

#Check again for more normal distribution

plt.subplots(figsize=(12,9))
sns.distplot(df_train['SalePrice'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(df_train['SalePrice'])

# plot with the distribution

plt.legend(['Normal Dist. ( mu = {:.2f} and sigma = {:.2f})'.format(mu, sigma)], loc = 'best')
plt.ylabel('Frequency')

# Probablity plot

fig = plt.figure()
stats.probplot(df_train['SalePrice'], plot = plt)
plt.show()

# Check the missing values
# Let's check if the data set has any missing values

df_train.columns[df_train.isnull().any()]

# plot of missing value attributes
plt.figure(figsize=(12,9))
sns.heatmap(df_train.isnull())
plt.show()

# Count Missing Values

# dt = df_train.isnull().sum()

dt = df_train.isnull().sum()/len(df_train)*100
dt = dt[dt>0]
dt.sort_values(inplace=True, ascending=False)
dt

# Visualising missing values

# Convert into dataframe
dt = pd.DataFrame(dt)
dt.columns = ['Count']
dt.index.names = ['Name']
dt['Name'] = dt.index
dt

# Plot the Missing values
plt.figure(figsize=(13,5))
sns.set(style = 'whitegrid')
sns.barplot(x = 'Name', y = 'Count', data = dt)
plt.xticks(rotation = 90)
plt.show()

# Correlation between train attributes

# Separate variable into new dataframe from original dataframe which has only numerical values
df_train_corr = df_train.select_dtypes(include= [np.number])
df_train_corr.shape

# Delete Id because that is not need for corralation plot
del df_train_corr['Id']

# Coralation plot
corr = df_train_corr.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr, annot=True)

# Top 50% Corralation train attributes with sale-price
top_feature = corr.index[abs(corr['SalePrice']>0.5)]
plt.subplots(figsize=(12,12))
top_corr = df_train[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show() 

# Here OverallQual is highly correlated with target feature of saleprice by 82%

# unique value of OverallQual
dt = df_train.OverallQual.unique() 

sns.barplot(df_train.OverallQual, df_train.SalePrice)

# Boxplot
plt.figure(figsize=(18,8))
sns.boxplot(x=df_train.OverallQual,y=df_train.SalePrice)

col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sns.set(style='ticks')
sns.pairplot(df_train[col],size=3, kind='reg')

print("Find most important features relative to target")
corr = df_train.corr()
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
corr.SalePrice

# Imputting missing values

# Count Null Values
dt = df_train.isnull().sum()
dt.sort_values(ascending = False, inplace = False)

# Merge the Dataset Train and Test
whole_df = pd.concat([df_train, df_test], sort = False)
whole_df.shape
whole_df.head()

dt = whole_df.isnull().sum().sort_values(ascending=False)
dt = dt[dt > 0]
dt

# Missing Value of PoolQC
# PoolQC has missing value ratio is 99%+. So, there is fill by None
whole_df['PoolQC'] = whole_df['PoolQC'].fillna('None')

#Arround 50% missing values attributes have been fill by None

# Missing Value of MiscFeature
whole_df[["MiscFeature", "Id"]].groupby(['MiscFeature'], as_index=False).count()
whole_df['MiscFeature'] = whole_df['MiscFeature'].fillna('None')

# Missing Value of Alley
whole_df[["Alley", "Id"]].groupby(['Alley'], as_index=False).count()
whole_df['Alley'] = whole_df['Alley'].fillna('None')

# Missing Value of Fence
whole_df[["Fence", "Id"]].groupby(['Fence'], as_index=False).count()
whole_df['Fence'] = whole_df['Fence'].fillna('None')

# Missing Value of FireplaceQu
whole_df[["FireplaceQu", "Id"]].groupby(['FireplaceQu'], as_index=False).count()
whole_df['FireplaceQu'] = whole_df['FireplaceQu'].fillna('None')

# Missing Value of FireplaceQu
whole_df[["FireplaceQu", "Id"]].groupby(['FireplaceQu'], as_index=False).count()
whole_df['FireplaceQu'] = whole_df['FireplaceQu'].fillna('None')

# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
whole_df[["LotFrontage", "Id"]].groupby(['LotFrontage'], as_index=False).count()
whole_df['LotFrontage'] = whole_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# Missing Value of GarageType
whole_df[["GarageType", "Id"]].groupby(['GarageType'], as_index=False).count()
whole_df['GarageType'] = whole_df['GarageType'].fillna('Attchd')

# Missing Value of GarageFinish
whole_df[["GarageFinish", "Id"]].groupby(['GarageFinish'], as_index=False).count()
whole_df['GarageFinish'] = whole_df['GarageFinish'].fillna('Unf')

# Missing Value of GarageQual
whole_df[["GarageQual", "Id"]].groupby(['GarageQual'], as_index=False).count()
whole_df['GarageQual'] = whole_df['GarageQual'].fillna('TA')

# Missing Value of GarageCond
whole_df[["GarageCond", "Id"]].groupby(['GarageCond'], as_index=False).count()
whole_df['GarageCond'] = whole_df['GarageCond'].fillna('TA')

# Missing Value of BsmtFinType2
whole_df[["BsmtFinType2", "Id"]].groupby(['BsmtFinType2'], as_index=False).count()
whole_df['BsmtFinType2'] = whole_df['BsmtFinType2'].fillna('Unf')

# Missing Value of BsmtExposure
whole_df[["BsmtExposure", "Id"]].groupby(['BsmtExposure'], as_index=False).count()
whole_df['BsmtExposure'] = whole_df['BsmtExposure'].fillna('No')

# Missing Value of BsmtFinType1
whole_df[["BsmtFinType1", "Id"]].groupby(['BsmtFinType1'], as_index=False).count()
whole_df['BsmtFinType1'] = whole_df['BsmtFinType1'].fillna('Unf')

# Missing Value of BsmtCond
whole_df[["BsmtCond", "Id"]].groupby(['BsmtCond'], as_index=False).count()
whole_df['BsmtCond'] = whole_df['BsmtCond'].fillna('TA')

# Missing Value of BsmtQual
whole_df[["BsmtQual", "Id"]].groupby(['BsmtQual'], as_index=False).count()
whole_df['BsmtQual'] = whole_df['BsmtQual'].fillna('TA')

dt = whole_df.isnull().sum()
dt = dt[dt>0]
dt.sort_values(ascending=False, inplace=True)
dt

# Missing Value of MasVnrType 
whole_df[["MasVnrType", "Id"]].groupby(['MasVnrType'], as_index=False).count()
whole_df['MasVnrType'] = whole_df['MasVnrType'].fillna('None')

# Missing Value of MasVnrArea
whole_df[["MasVnrArea", "Id"]].groupby(['MasVnrArea'], as_index=False).count()
whole_df['MasVnrArea'] = whole_df['MasVnrArea'].fillna(whole_df.MasVnrArea.mean())

# Missing Value of MSZoning
whole_df[["MSZoning", "Id"]].groupby(['MSZoning'], as_index=False).count()
whole_df['MSZoning'] = whole_df['MSZoning'].fillna('RL')

# Missing Value of Functional
whole_df[["Functional", "Id"]].groupby(['Functional'], as_index=False).count()
whole_df['Functional'] = whole_df['Functional'].fillna('Typ')

# Missing Value of BsmtHalfBath
whole_df[["BsmtHalfBath", "Id"]].groupby(['BsmtHalfBath'], as_index=False).count()
whole_df['BsmtHalfBath'] = whole_df['BsmtHalfBath'].fillna(whole_df.BsmtHalfBath.median())

# Missing Value of BsmtFullBath
whole_df[["BsmtFullBath", "Id"]].groupby(['BsmtFullBath'], as_index=False).count()
whole_df['BsmtFullBath'] = whole_df['BsmtFullBath'].fillna(whole_df.BsmtFullBath.median())

# Missing Value of Utilities
whole_df[["Utilities", "Id"]].groupby(['Utilities'], as_index=False).count()
whole_df['Utilities'] = whole_df['Utilities'].fillna('AllPub')

# Missing Value of KitchenQual
whole_df[["KitchenQual", "Id"]].groupby(['KitchenQual'], as_index=False).count()
whole_df['KitchenQual'] = whole_df['KitchenQual'].fillna('TA')

# Missing Value of Electrical 
whole_df[["Electrical", "Id"]].groupby(['Electrical'], as_index=False).count()
whole_df['Electrical'] = whole_df['Electrical'].fillna('SBrkr')

# Missing Value of SaleType 
whole_df[["SaleType", "Id"]].groupby(['SaleType'], as_index=False).count()
whole_df['SaleType'] = whole_df['SaleType'].fillna('WD')

# Missing Value of BsmtUnfSF 
whole_df[["BsmtUnfSF", "Id"]].groupby(['BsmtUnfSF'], as_index=False).count()
whole_df['BsmtUnfSF'] = whole_df['BsmtUnfSF'].fillna(whole_df.BsmtUnfSF.mean())

# Missing Value of BsmtFinSF2 
whole_df[["BsmtFinSF2", "Id"]].groupby(['BsmtFinSF2'], as_index=False).count()
whole_df['BsmtFinSF2'] = whole_df['BsmtFinSF2'].fillna(whole_df.BsmtFinSF2.mean())

# Missing Value of BsmtFinSF1 
whole_df[["BsmtFinSF1", "Id"]].groupby(['BsmtFinSF1'], as_index=False).count()
whole_df['BsmtFinSF1'] = whole_df['BsmtFinSF1'].fillna(whole_df.BsmtFinSF1.mean())

# Missing Value of GarageCars 
whole_df[["GarageCars", "Id"]].groupby(['GarageCars'], as_index=False).count()
whole_df['GarageCars'] = whole_df['GarageCars'].fillna(whole_df.GarageCars.median())

# Missing Value of GarageArea 
whole_df[["GarageArea", "Id"]].groupby(['GarageArea'], as_index=False).count()
whole_df['GarageArea'] = whole_df['GarageArea'].fillna(whole_df.GarageArea.mean())

# Missing Value of Exterior2nd 
whole_df[["Exterior2nd", "Id"]].groupby(['Exterior2nd'], as_index=False).count()
whole_df['Exterior2nd'] = whole_df['Exterior2nd'].fillna('VinylSd')

# Missing Value of Exterior1st 
whole_df[["Exterior1st", "Id"]].groupby(['Exterior1st'], as_index=False).count()
whole_df['Exterior1st'] = whole_df['Exterior1st'].fillna('VinylSd')

# Infer Missing Values

# Missing Value of TotalBsmtSF
whole_df['TotalBsmtSF'].fillna(whole_df['1stFlrSF'], inplace=True)

# Missing Value of GarageYrBlt
whole_df['GarageYrBlt'].fillna(whole_df['YearBuilt'], inplace=True)

#  Checking there is any null value or not

# dt = whole_df.isnull().sum()
# dt = dt[dt>0]
# dt.sort_values(ascending=False, inplace=True)
# dt

plt.figure(figsize=(10, 5))
sns.heatmap(whole_df.isnull())

# Encoding str to int

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond','ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope','LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 
        'OverallCond','YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood','Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 
        'RoofMatl', 'Exterior1st','Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature','SaleType', 'SaleCondition', 'Electrical', 
        'Heating', 'Utilities')

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

for c in cols:
  whole_df[c] = label.fit_transform(list(whole_df[c].values))
  
# Prepraring data for prediction
  
# Time to split the database back into two parts, one with sales price and one without

train_X = whole_df[whole_df['SalePrice'].notnull()]
train_X.shape

# del train_X['SalePrice']

test_X = whole_df[whole_df['SalePrice'].isnull()]
test_X.shape

del test_X['SalePrice']

# Prepraring data for prediction

# Take their values in X and y
X = train_X[[col for col in train_X.columns if col!='SalePrice']]
# y = whole_df['SalePrice'][whole_df['SalePrice']>0]
y = train_X['SalePrice']
Test_X = test_X

# Split data into train and test formate

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)

# Linear Regression

# Train the model
from sklearn import linear_model
model = linear_model.LinearRegression()

# Fit the Model
model.fit(X_train,Y_train)

# Prediction
print("Predict value " + str(model.predict(X_test)))
print("Real value " + str(Y_test))

# Score/Accuracy
print("Accuracy --> ", model.score(X_test, Y_test)*100)

# Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

# Train the Model
model = RandomForestRegressor(n_estimators=1000)

# Fit
model.fit(X_train, Y_train)

# Score/Accuracy
print("Accuracy-------> ", model.score(X_test, Y_test)*100)

# Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor

# Train Model
GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)

# Fit
GBR.fit(X_train, Y_train)

# Score/Accuracy
print("Accuracy--------> ", GBR.score(X_test, Y_test)*100)

id = test_X.Id
result = model.predict(test_X)

output = pd.DataFrame({'Id' : id , 'SalePrice' : result})
output = output[['Id', 'SalePrice']]

output.to_csv("gdrive/My Drive/Colab Notebooks/House Cost Prediction/Solution.csv", index=False)
output.head()
