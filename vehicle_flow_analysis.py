import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic vehicle flow data
def generate_vehicle_data():
    hours = np.arange(24)
    base_flow = 50
    peak_morning = 8
    peak_evening = 18
    vehicle_counts = []
    temperatures = []
    time_labels = []
    weather = []

    for hour in hours:
        flow = base_flow

        if 6 <= hour <= 9:
            flow += 80 * np.exp(-0.5 * ((hour - peak_morning) / 1.5) ** 2)

        if 17 <= hour <= 20:
            flow += 100 * np.exp(-0.5 * ((hour - peak_evening) / 1.5) ** 2)

        if 10 <= hour <= 16:
            flow += 30 + 10 * np.sin((hour - 10) * np.pi / 6)

        if hour >= 22 or hour <= 5:
            flow *= 0.3

        flow += (np.random.rand() - 0.5) * 20
        flow = max(0, round(flow))

        temp = 20 + 15 * np.sin((hour - 6) * np.pi / 12) + (np.random.rand() - 0.5) * 5
        time_label = f"{hour:02d}:00"
        if 12 <= hour <= 18:
            weather_type = "Sunny"
        elif 6 <= hour <= 11:
            weather_type = "Morning"
        else:
            weather_type = "Night"

        vehicle_counts.append(flow)
        temperatures.append(temp)
        time_labels.append(time_label)
        weather.append(weather_type)

    df = pd.DataFrame({
        "hour": hours,
        "vehicle_count": vehicle_counts,
        "temperature": temperatures,
        "time_label": time_labels,
        "weather": weather
    })

    return df

# Perform linear regression
def linear_regression(x, y):
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    y_pred = slope * x + intercept
    return slope, intercept, y_pred

# Calculate correlation and R^2
def correlation_and_r2(y_true, y_pred):
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return correlation, r_squared

# Main execution
df = generate_vehicle_data()
x_hour = df["hour"]
y_vehicles = df["vehicle_count"]
x_temp = df["temperature"]

# Hour vs Vehicle Count Regression
slope_hour, intercept_hour, pred_hour = linear_regression(x_hour, y_vehicles)
corr_hour, r2_hour = correlation_and_r2(y_vehicles, pred_hour)

# Temperature vs Vehicle Count Regression
slope_temp, intercept_temp, pred_temp = linear_regression(x_temp, y_vehicles)
corr_temp, r2_temp = correlation_and_r2(y_vehicles, pred_temp)

# Predictions for next 6 hours
future_hours = np.arange(24, 30)
predicted_vehicles = slope_hour * (future_hours % 24) + intercept_hour
future_time_labels = [f"{(h % 24):02d}:00" for h in future_hours]
predictions = pd.DataFrame({
    "Hour": future_hours,
    "Time": future_time_labels,
    "Predicted Vehicles": predicted_vehicles.round().astype(int)
})

# Summary
summary = {
    "Average Vehicles": round(np.mean(y_vehicles)),
    "Max Vehicles": int(np.max(y_vehicles)),
    "Min Vehicles": int(np.min(y_vehicles)),
    "Peak Hour": int(x_hour[np.argmax(y_vehicles)]),
    "Lowest Hour": int(x_hour[np.argmin(y_vehicles)]),
    "R² (Hour vs Vehicles)": round(r2_hour, 4),
    "R² (Temp vs Vehicles)": round(r2_temp, 4),
    "Correlation (Hour)": round(corr_hour, 4),
    "Correlation (Temp)": round(corr_temp, 4),
    "Regression Equation (Hour)": f"y = {slope_hour:.2f}x + {intercept_hour:.2f}",
    "Regression Equation (Temp)": f"y = {slope_temp:.2f}x + {intercept_temp:.2f}"
}

# Display summary
print("\n--- Vehicle Flow Analysis Summary ---")
for key, value in summary.items():
    print(f"{key}: {value}")

# Display predictions
print("\n--- Predictions for Next 6 Hours ---")
print(predictions)

# Plot actual vs predicted (Hour)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(df["time_label"], y_vehicles, label="Actual", marker='o')
plt.plot(df["time_label"], pred_hour, label="Predicted (Hour Regression)", linestyle='--')
plt.title("Vehicle Flow vs Time")
plt.xlabel("Hour")
plt.ylabel("Vehicle Count")
plt.xticks(rotation=45)
plt.legend()

# Plot temperature vs vehicle count with regression
plt.subplot(1, 2, 2)
sns.scatterplot(x="temperature", y="vehicle_count", data=df)
sns.lineplot(x=x_temp, y=pred_temp, color='red', label='Regression Line')
plt.title("Temperature vs Vehicle Count")
plt.xlabel("Temperature (°C)")
plt.ylabel("Vehicle Count")
plt.legend()

plt.tight_layout()
plt.show()
