import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset
np.random.seed(42)

# Room types
room_types = ['Single', 'Double', 'Suite']

# Generate synthetic data
num_samples = 1000
room_sizes = np.random.randint(20, 100, size=num_samples)  # Room sizes between 20 and 100 square meters
room_types_data = np.random.choice(room_types, size=num_samples)

# Generate room rates based on size and type
room_rates = room_sizes * np.random.uniform(30, 60, size=num_samples)  # Base rate per square meter
room_rates += np.where(room_types_data == 'Single', 50, 0)
room_rates += np.where(room_types_data == 'Double', 100, 0)
room_rates += np.where(room_types_data == 'Suite', 200, 0)

# Create DataFrame
df = pd.DataFrame({
    'RoomSize': room_sizes,
    'RoomType': room_types_data,
    'RoomRate': room_rates
})

# Convert RoomType to categorical data
df['RoomType'] = df['RoomType'].astype('category').cat.codes

# Split data into features and target
X = df[['RoomSize', 'RoomType']]
y = df['RoomRate']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree Regressor model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
