#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[153]:


df = pd.read_csv("D:\\ML project\\life expectancy\\Life Expectancy Data.csv")


# In[32]:


df.tail()


# In[6]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[154]:


df.columns = df.columns.str.strip()


# In[34]:


df['Status'].unique()


# In[27]:


sns.boxplot(x=df["Life expectancy"])


# In[28]:


sns.countplot(x=df['Life expectancy'])


# In[35]:


developed_df = df[df['Status']=='Developed']
sns.countplot(x=developed_df['Life expectancy'])


# In[36]:


sns.countplot(x=df['Status'])


# In[48]:


avg_life_expectancy_by_year = df.groupby('Year')['Life expectancy'].mean().reset_index()
sns.lineplot(avg_life_expectancy_by_year,x='Year',y='Life expectancy')


# In[50]:


avg_life_expectancy_by_adult_mortatilty = df.groupby('Status')['Adult Mortality'].mean().reset_index()
sns.barplot(avg_life_expectancy_by_adult_mortatilty, x = 'Status',y = "Adult Mortality")


# In[52]:


df.isnull().sum()


# In[53]:


df.head()


# In[155]:


df.drop(['Population','Hepatitis B'],axis=1,inplace=True)


# In[57]:


df.head()


# In[61]:


df.isnull().sum()


# In[156]:


df.dropna(subset=['Life expectancy'],inplace=True)


# In[157]:


df.drop(['Country'],axis=1,inplace=True)


# In[158]:


df.drop(['thinness 5-9 years'],axis=1,inplace=True)


# In[159]:


df['Status'].replace({'Developed': 1,'Developing':0},inplace=True)


# In[87]:


plt.figure(figsize=(10,10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')


# In[72]:


df.describe()


# In[160]:


df['Alcohol'].fillna(df['Alcohol'].median(),inplace=True)
df['BMI'].fillna(df['BMI'].std(),inplace=True)
df['Polio'].fillna(df['Polio'].mean(),inplace=True)
df['Total expenditure'].fillna(df['Total expenditure'].mean(),inplace=True)
df['Diphtheria'].fillna(df['Diphtheria'].mean(),inplace=True)
#df['thinness 1-19 years'].fillna(df['thinness 1-19 years'].mean(),inplace=True)
df['Income composition of resources'].fillna(df['Income composition of resources'].mean(),inplace=True)
df['Schooling'].fillna(df['Schooling'].mean(),inplace=True)


# In[91]:


df.head()


# In[92]:


df.columns


# In[161]:


df['thinness  1-19 years'].fillna(df['thinness  1-19 years'].mean(),inplace=True)


# In[94]:


df.isnull().sum()


# In[162]:


df['GDP'].fillna(df['GDP'].mean(),inplace=True)


# In[144]:


df.isnull().sum()


# In[98]:


df.info()


# In[166]:


from sklearn.model_selection import train_test_split
X = df.drop(['Life expectancy'],axis=1)
y = df['Life expectancy']


# In[172]:


from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

# Identify numeric columns
num_features = X.select_dtypes(exclude='object').columns

# Create the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('StandardScaler', StandardScaler(), num_features)
    ],
    remainder='passthrough'
)

# ✅ Now fit_transform the data
X_transformed = preprocessor.fit_transform(X)

# ✅ Now access feature names (after fit)
scaled_feature_names = preprocessor.named_transformers_['StandardScaler'].get_feature_names_out(num_features)

# Get categorical columns that were passed through
cat_features = X.select_dtypes(include='object').columns

# Combine all column names
all_columns = list(scaled_feature_names) + list(cat_features)

# Convert to DataFrame
X = pd.DataFrame(X_transformed, columns=all_columns)
X


# In[164]:


# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# numeric_scalar = StandardScaler()
# num_features = X.select_dtypes(exclude='object').columns
# preprocessor = ColumnTransformer(
# [('StandardScaler',numeric_scalar,num_features)],remainder='passthrough'
# )


# In[165]:


# # Save the feature names
# scaled_features = preprocessor.named_transformers_['StandardScaler'].get_feature_names_out(num_features)
# all_features = list(scaled_features) + list(X.select_dtypes(include='object').columns)

# # Apply transform and convert to DataFrame
# X = pd.DataFrame(preprocessor.fit_transform(X), columns=all_features)


# In[147]:


# X = preprocessor.fit_transform(X)


# In[148]:


# pd.DataFrame(X)


# In[173]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[174]:


from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[175]:


def evaluate_model(true,predicted):
    mae = mean_absolute_error(true,predicted)
    mse = mean_squared_error(true,predicted)
    rmse = np.sqrt(mean_squared_error(true,predicted))
    r2_square = r2_score(true,predicted)
    return mae,rmse,r2_square


# In[177]:


models = {
    "Svr":SVR(),
#    "KNN":KNeighborsRegressor(),
    "SGD":SGDRegressor(),
    "RandomForest":RandomForestRegressor(),
    "Linear Regression":LinearRegression(),
    "Ridge":Ridge(),
    "Lasso":Lasso(),
    "Decision Tree":DecisionTreeRegressor()
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train,y_train)
    
    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)
    
    model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train,y_train_preds)
    model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test,y_test_preds)
    
    print(list(models.keys())[i])
    
    print('Model performance for training set')
    print('RMSE: {:.4f}'.format(model_train_rmse))
    print('MAE: {:.4f}'.format(model_train_mae))
    print('R2 Score: {:.4f}'.format(model_train_r2))
    
    print('-------------------------------------------')
    
    print('Model performance for test set')
    print('RMSE: {:.4f}'.format(model_test_rmse))
    print('MAE: {:.4f}'.format(model_test_mae))
    print('R2 Score: {:.4f}'.format(model_test_r2))
    
    print('='*35)
    print('\n')
    


# In[ ]:




