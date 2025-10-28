import pandas as pd
import matplotlib.pyplot as plt

# Load the sales data from CSV file
sales_data = pd.read_csv("/content/customer_shopping_data.csv", engine='python', on_bad_lines='skip')
print("CSV Data Preview:")
print(sales_data.head())

print("\nData Information:")
print(sales_data.info())

# Check for duplicate rows in the CSV data
print("\nDuplicate rows in CSV data:", sales_data.duplicated().sum())
# Check for missing values in the CSV data
print("\nMissing values in CSV data:\n", sales_data.isnull().sum())

# Remove duplicates
sales_data = sales_data.drop_duplicates()

sales_data['price'] = pd.to_numeric(sales_data['price'], errors='coerce').fillna(0)

# Fill missing values for 'quantity' with 0, as quantity is likely a numerical value
sales_data['quantity'] = pd.to_numeric(sales_data['quantity'], errors='coerce').fillna(0)
# Fill missing values 'Unknown'
sales_data['payment_method'] = sales_data['payment_method'].fillna('Unknown')
sales_data['shopping_mall'] = sales_data['shopping_mall'].fillna('Unknown')
sales_data['invoice_date'] = sales_data['invoice_date'].fillna('Unknown')

# Check for missing values after filling
print("\nMissing values after filling:\n", sales_data.isnull().sum())

# Additional data merging from other files, if available
df_file1 = pd.read_csv("/content/customer_shopping_data.csv", engine='python', on_bad_lines='skip')
df_file2 = pd.read_csv("/content/customer_shopping_data1.csv", engine='python', on_bad_lines='skip')
merged_df = pd.concat([df_file1, df_file2], ignore_index=True)
print("\nMerged DataFrame:")
print(merged_df.head())

# Data Transformation
merged_df[['Year', 'Month', 'Day']] = merged_df['invoice_date'].str.split('/', expand=True)

# Total sales, average order value, product category distribution
total_sales = sales_data['price'].sum()
average_order_value = sales_data['price'].mean()
product_category_distribution = sales_data['category'].value_counts(normalize=True)

print(f"\nTotal Sales: {total_sales}")
print(f"Average Order Value: {average_order_value:.2f}")
print("\nProduct Category Distribution:\n", product_category_distribution)

# Check unique values in the 'category' column
print("\nUnique values in 'category' column:\n", sales_data['category'].unique())

# Aggregated sales by product type
desired_categories = ['Clothing', 'Shoes', 'Books', 'Cosmetics', 'Food & Beverage', 'Toys', 'Technology', 'Souvenir', 'Cosmnyon']
filtered_sales_data = sales_data[sales_data['category'].isin(desired_categories)]
grouped_sales = filtered_sales_data.groupby('category')['price'].sum()

# Bar plot: Total Sales by Product Type
plt.figure(figsize=(10, 6))
grouped_sales.plot(kind='bar', color='skyblue')
plt.title('Total Sales by Product Type (Selected Categories)')
plt.xlabel('Product Type')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie chart: Product Category Distribution
plt.figure(figsize=(8, 8))
product_category_distribution = filtered_sales_data['category'].value_counts()
plt.pie(product_category_distribution, labels=product_category_distribution.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('Product Category Distribution (Selected Categories)')
plt.tight_layout()
plt.show()

