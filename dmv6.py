import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("customer_shopping_data.csv") 
df.head()

df.tail()

# To check the count of records grouped by region/branch of the mall
df.groupby("shopping_mall").count()

# To check the count of records grouped by the product categories
df.groupby("category").count()

# total sales for each mall branch
branch_sales = df.groupby("shopping_mall").sum()
branch_sales

# total sales for each category of product
category_sales = df.groupby("category").sum()
category_sales

# to get the top performing branches
branch_sales.sort_values(by = "price", ascending = False)

# to get the top selling categories
category_sales.sort_values(by = "price", ascending = False)

# to get total sales for each combination of branch and product_category
combined_branch_category_sales = df.groupby(["shopping_mall", "category"]).sum()
combined_branch_category_sales

# pie chart for sales by branch
plt.pie(branch_sales["price"], labels = branch_sales.index) 
plt.show()

# pie chart for sales by product category
plt.pie(category_sales["price"], labels = category_sales.index) 
plt.show()

combined_pivot = df.pivot_table(index="shopping_mall", columns="category", values="price", aggfunc="sum") 
# grouped bar chart for sales of different categories at different branches
combined_pivot.plot(kind="bar", figsize=(10, 6)) 
plt.show()


