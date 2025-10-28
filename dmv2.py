import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Fetching Weather Data
API_KEY = '3feca6bf43580bfaa2aa7edb85929cf1'
CITY = 'Pune'
BASE_URL = 'http://api.openweathermap.org/data/2.5/weather'

params = {
    'q': CITY,
    'appid': API_KEY,
    'units': 'metric'  # Use 'imperial' for Fahrenheit
}

response = requests.get(BASE_URL, params=params)
weather_data = response.json()
# Extract weather details
temperature = weather_data['main']['temp']
humidity = weather_data['main']['humidity']
wind_speed = weather_data['wind']['speed']
precipitation = weather_data.get('rain', {}).get('1h', 0)

# Storing in DataFrame
data = {
    'temperature': [temperature],
    'humidity': [humidity],
    'wind_speed': [wind_speed],
    'precipitation': [precipitation]
}

df = pd.DataFrame(data)
# Handle missing values
df.fillna(0, inplace=True)

# Display weather data statistics
average_temperature = df['temperature'].mean()
max_temperature = df['temperature'].max()
min_temperature = df['temperature'].min()
print(f'Avg: {average_temperature}, Max: {max_temperature}, Min: {min_temperature}')

# Plot temperature data as bar plot
plt.figure(figsize=(10, 5))
plt.bar(['Temperature'], [temperature], color='blue')
plt.title('Temperature')
plt.ylabel('Temperature (°C)')
plt.show()

dates = pd.date_range(start='2024-01-01', periods=5, freq='D')

df_time = pd.DataFrame({
    'date': dates,
    'temperature': [temperature + i for i in range(5)],      # Incremental rise
    'humidity': [humidity - i*2 for i in range(5)],           # Decremental drop
    'wind_speed': [wind_speed + i*0.2 for i in range(5)],     # Small rise
    'precipitation': [precipitation + i*0.05 for i in range(5)]  # Incremental rise
})

print(df_time.head())
# Aggregate by date and calculate average temperature
daily_avg = df_time.groupby(df_time['date'].dt.date).mean()
print("Daily Average Temperature:", daily_avg['temperature'])

# Plot daily average temperature
daily_avg['temperature'].plot(kind='line', title='Daily Average Temperature', xlabel='Date', ylabel='Temperature (°C)')
plt.show()

if 'coord' in weather_data:
    latitude = weather_data['coord']['lat']
    longitude = weather_data['coord']['lon']
    print(f"Coordinates: Latitude {latitude}, Longitude {longitude}")

    # Using Plotly for a simple map plot
    map_fig = px.scatter_geo(
        lat=[latitude],
        lon=[longitude],
        text=[f"City: {CITY}\nTemperature: {temperature}°C\nHumidity: {humidity}%"],
        title="Weather Location",
    )
    map_fig.update_layout(geo_scope="asia")  # Centers the map around Asia
    map_fig.show()

correlation_matrix = df_time[['temperature', 'humidity', 'wind_speed', 'precipitation']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Weather Attributes")
plt.show()
