import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the dataset
data = pd.read_csv('City_Air_Quality.csv')

# Explore the dataset
print(data.head())
print(data.info())

# Assuming columns are named as 'Date', 'PM2.5', 'PM10', 'CO', and 'AQI'
data['Date'] = pd.to_datetime(data['Date'])

# Create a line plot to visualize the overall AQI trend over time
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['AQI'], label='AQI', color='blue')
plt.xlabel('Date')
plt.ylabel('Air Quality Index (AQI)')
plt.title('AQI Trend Over Time')
plt.legend()
plt.grid(True)
plt.show()


# Plot individual pollutant levels on separate line plots
# Plot for PM2.5 levels
plt.figure(figsize=(10,6))
plt.plot(data['Date'], data['PM2.5'], color='green', label='PM2.5')
plt.title('PM2.5 Trend Over Time')
plt.xlabel('Date')
plt.ylabel('PM2.5 Levels')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Repeat for PM10 and CO
plt.figure(figsize=(10,6))
plt.plot(data['Date'], data['PM10'], color='orange', label='PM10')
plt.title('PM10 Trend Over Time')
plt.xlabel('Date')
plt.ylabel('PM10 Levels')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(data['Date'], data['CO'], color='red', label='CO')
plt.title('CO Trend Over Time')
plt.xlabel('Date')
plt.ylabel('CO Levels')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Use bar plots to compare AQI values across different dates
plt.figure(figsize=(12, 3))
data['Month'] = data['Date'].dt.to_period('M')
avg_aqi_per_month = data.groupby('Month')['AQI'].mean().reset_index()
plt.bar(avg_aqi_per_month['Month'].astype(str), avg_aqi_per_month['AQI'], color='orange')
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Average AQI')
plt.title('Average AQI per Month')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=data[['PM2.5', 'PM10', 'CO']])
plt.title('Distribution of Pollutant Levels')
plt.ylabel('Concentration')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(data['PM2.5'], data['AQI'], label='PM2.5', alpha=0.5, color='red')
plt.scatter(data['PM10'], data['AQI'], label='PM10', alpha=0.5, color='green')
plt.scatter(data['CO'], data['AQI'], label='CO', alpha=0.5, color='blue')
plt.xlabel('Pollutant Level')
plt.ylabel('Air Quality Index (AQI)')
plt.title('Relationship between AQI and Pollutant Levels')
plt.legend()
plt.grid(True)
plt.show()

#################
data['Date'].fillna(method='ffill', inplace=True)
data['Time'].fillna(method='ffill', inplace=True)
data.isnull().sum()

top10=sns.barplot(x='CO(GT)', y='Time', data=data[:10], color='red', ci=None)
top10.invert_yaxis()

sns.lineplot(x='CO(GT)', y="Time", data=data[:10], color='navy', ci=None)

sns.boxplot(x='CO(GT)', data=data[:1000], color='navy')
plt.xlabel("CO(GT)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
