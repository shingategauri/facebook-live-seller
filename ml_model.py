import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv("Live_20210128.csv")

# Drop empty columns
df = df.drop(columns=["Column1", "Column2", "Column3", "Column4"])

# Drop rows with missing values in key columns
df = df.dropna(subset=["status_type", "num_shares"])

# One-hot encode the 'status_type' column
df = pd.get_dummies(df, columns=["status_type"], drop_first=True)

# Define features and target
X = df.drop(columns=["status_id", "status_published", "num_shares"])
y = df["num_shares"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")
