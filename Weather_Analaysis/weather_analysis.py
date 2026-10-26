import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("weather.csv")

print("Weather Data:")
print(data)

print("\nAverage Temperature:", data["Temperature"].mean())
print("Maximum Temperature:", data["Temperature"].max())
print("Minimum Temperature:", data["Temperature"].min())

highest_temp = data["Temperature"].max()

hottest_day = data[data["Temperature"] == highest_temp]

print("\nDay with Highest Temperature:")
print(hottest_day[["Date", "Temperature"]])

plt.plot(data["Date"], data["Temperature"], marker="o")

plt.title("Daily Temperature Changes")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.hist(data["Temperature"], bins=5, edgecolor="black")

plt.title("Distribution of Temperatures")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.show()

plt.scatter(data["Humidity"], data["Temperature"])

plt.title("Humidity vs Temperature")
plt.xlabel("Humidity")
plt.ylabel("Temperature")
plt.show()
