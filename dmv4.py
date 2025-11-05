import pandas as pd
import matplotlib.pyplot as plt 

df  = pd.read_csv('RealEstate_price.csv')
df.head()

df.shape

df.size

df.info()

df['Price'].isnull().sum()

df['Price'] = df['Price'].fillna(df['Price'].mean())

df['Price'].isnull().sum()

df.isnull().sum()

df.duplicated().sum()

df.info()

df['Sales_date'] = pd.to_datetime(df['Sales_date'])

df.info()

# basic filtering accordinf to need 
df.head()

df[df['Price']>100000]  

df[df['Bedrooms'] > 3]

east_houses = df[df['Neighborhood'] == 'East']
east_houses

from sklearn.preprocessing import LabelEncoder

# Identify categorical columns
cat_cols = df.select_dtypes(include='object').columns
print("\nCategorical Columns:", cat_cols.tolist())

# Label encode binary columns, one-hot encode multi-category columns
le = LabelEncoder()
for col in cat_cols:
    if df[col].nunique() == 2:
        df[col] = le.fit_transform(df[col])

# Average price by number of bedrooms
avg_price_by_bedrooms = df.groupby('Bedrooms')['Price'].mean()
print("\nAverage Sale Price by Bedrooms:")
print(avg_price_by_bedrooms)

avg_price_by_neighborhood = df.groupby('Neighborhood')['Price'].mean().sort_values(ascending=False)
print("\nAverage Sale Price by Neighborhood:")
print(avg_price_by_neighborhood)

df.head()

import matplotlib.pyplot as plt

num_cols = ['Price', 'SqFt', 'Bedrooms', 'Bathrooms', 'Offers']

# Visualize distributions
for col in num_cols:
    plt.figure(figsize=(6,4))
    plt.boxplot(df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

#Apply IQR-based capping
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df[col] = df[col].clip(lower, upper)
#after clipping 
import matplotlib.pyplot as plt

num_cols = ['Price', 'SqFt', 'Bedrooms', 'Bathrooms', 'Offers']

# Visualize distributions
for col in num_cols:
    plt.figure(figsize=(6,4))
    plt.boxplot(df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Export cleaned dataset for further use
df.to_csv("Cleaned_RealEstate_Prices.csv", index=False)
print("\nCleaned dataset saved as 'Cleaned_RealEstate_Prices.csv'")
