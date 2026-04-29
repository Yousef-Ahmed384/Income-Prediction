import pandas as pd
imp
ort numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

dataFrame=pd.read_csv("train_data.csv",na_values=['?'],skipinitialspace=True)
dataFrame.head(30)
print(dataFrame.duplicated().sum())
df_no_duplicates = dataFrame.drop_duplicates()
df_no_duplicates=df_no_duplicates.drop(columns='education')

mask = df_no_duplicates.isnull().any(axis=1)
print(mask.sum()/len(df_no_duplicates))
mask = df_no_duplicates.isnull().sum()
print(mask/len(df_no_duplicates))
print(len(df_no_duplicates))
df_cleaned = df_no_duplicates.dropna()
print(len(df_cleaned))
df_cleaned.head()
df = pd.get_dummies(df_cleaned, columns=["workclass", "occupation", "race","marital-status","native-country","relationship"],dtype=int)
df.head()
sex_dict = {'Male':1,'Female':0}
income_dict = { '>50K' : 1,'<=50K' :0}
print(df.columns)
df['sex'] = df['sex'].replace(sex_dict)
df['Income '] = df['Income '].replace(income_dict)
df
q1 = df['fnlwgt'].quantile(0.25)
q3 = df['fnlwgt'].quantile(0.75)
IQR = q3 - q1
lowerBound = q1 - 1.5 * IQR
upperBound = q3 + 1.5 * IQR
df['fnlwgt'] = df['fnlwgt'].clip(lower=lowerBound, upper=upperBound)
df
corr_data = df.corr()
top_feature = corr_data.index[abs(corr_data['Income ']) > 0.3]
plt.subplots(figsize=(12,8))
top_corr = df[top_feature].corr()
sns.heatmap(top_corr,annot=True)
plt.show()
top_feature = top_feature.delete(-1)

top_feature


data_input = df[top_feature]

data_input = data_input.drop(columns="Income ")
data_output = df['Income ']



x_train,x_val,y_train,y_val= train_test_split(data_input,data_output,test_size=0.2,random_state=0)

x_train
scalar = StandardScaler()
scalar.fit(x_train)
x_train_scaled = scalar.transform(x_train)
x_val_scaled = scalar.transform(x_val)

x_train_scaled = pd.DataFrame(x_train_scaled,columns=x_train.columns)
x_val_scaled = pd.DataFrame(x_val_scaled,columns=x_val.columns)

