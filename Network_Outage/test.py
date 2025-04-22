import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load datasets
outage_data = pd.read_csv('eaglei_outages_2014.csv')
coverage_data = pd.read_csv('coverage_history.csv')
modeled_customers_data = pd.read_csv('MCC.csv')
dqi_data = pd.read_csv('DQI.csv')

# Data Preprocessing
# Convert run_start_time to datetime (auto-detect format)
outage_data['run_start_time'] = pd.to_datetime(outage_data['run_start_time'])

# Extract features from the timestamp
outage_data['year'] = outage_data['run_start_time'].dt.year.astype(int)  # Ensure year is int
outage_data['month'] = outage_data['run_start_time'].dt.month

# Convert year in coverage_data from date string to int
coverage_data['year'] = pd.to_datetime(coverage_data['year'], format='%m/%d/%y').dt.year

# Ensure state is consistent across datasets
# Convert state names to abbreviations in outage_data
state_abbreviations = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    # Add other states as needed
}

# Replace full state names with their abbreviations in outage_data
outage_data['state'] = outage_data['state'].replace(state_abbreviations)

# Ensure all state columns are strings
outage_data['state'] = outage_data['state'].astype(str)
coverage_data['state'] = coverage_data['state'].astype(str)
dqi_data['fema'] = dqi_data['fema'].astype(str)  # Assuming 'fema' corresponds to state codes

# Convert fips_code to string for consistency
outage_data['fips_code'] = outage_data['fips_code'].astype(str)
modeled_customers_data['County_FIPS'] = modeled_customers_data['County_FIPS'].astype(str)

# Check and print data types for debugging
print("Data types before merging:")
print("Outage Data:", outage_data.dtypes)
print("Coverage Data:", coverage_data.dtypes)
print("MCC Data:", modeled_customers_data.dtypes)
print("DQI Data:", dqi_data.dtypes)

# Encode categorical variables for county
label_encoder = LabelEncoder()
outage_data['county'] = label_encoder.fit_transform(outage_data['county'])

# Merge coverage_data
data = outage_data.merge(coverage_data, left_on=['year', 'state'], right_on=['year', 'state'], how='left')
print("After merging with coverage_data:", data.columns)

# Merge modeled_customers_data
data = data.merge(modeled_customers_data, left_on='fips_code', right_on='County_FIPS', how='left')
print("After merging with modeled_customers_data:", data.columns)

# Merge dqi_data
data = data.merge(dqi_data, left_on=['year', 'state'], right_on=['year', 'fema'], how='left')
print("After merging with dqi_data:", data.columns)

data = data.rename(columns={'total_customers_x': 'total_customers'})  # Rename for clarity
data.drop(columns=['total_customers_y'], inplace=True)  # Drop the unwanted column

if 'county' in data.columns and 'state' in data.columns:
    # One-Hot Encoding for categorical variables (county and state)
    data = pd.get_dummies(data, columns=['county', 'state'], drop_first=True)
else:
    print("Warning: 'county' or 'state' not found in the DataFrame.")

# If 'total_customers' is missing, you can check for it before using it
if 'total_customers' not in data.columns:
    print("Warning: 'total_customers' column is not in the merged DataFrame.")

# Fill missing values if necessary
data.fillna(0, inplace=True)

print(data.dtypes)  # Check data types after merging

# Define features and target variable
X = data.drop(columns=['customers_out', 'run_start_time', 'fips_code']).values  # Exclude the target variable
y = data['customers_out'].values  # Target variable (number of customers without power)


# Check the contents of X before training
print("Features (X):", X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Calculate training predictions and accuracy
y_train_pred = model.predict(X_train)

# Calculate testing accuracy
test_r2 = r2_score(y_test, y_pred) * 100  # Convert to percentage

print("----------------------------------------------------------------------------------------------------")
# Print accuracies
print(f'Testing Accuracy: {test_r2:.2f}%')

# Function to predict outages for new data
def predict_outages(new_data):
    # Preprocess new_data similarly
    new_data['run_start_time'] = pd.to_datetime(new_data['run_start_time'])  # Automatically infers the format
    new_data['year'] = new_data['run_start_time'].dt.year.astype(int)  # Ensure year is int
    new_data['month'] = new_data['run_start_time'].dt.month

    # Replace full state names with their abbreviations if necessary
    new_data['state'] = new_data['state'].replace(state_abbreviations)

    # Ensure state and county are treated as strings
    new_data['state'] = new_data['state'].astype(str)
    new_data['county'] = label_encoder.transform(new_data['county'])

    # Convert fips_code to string
    new_data['fips_code'] = new_data['fips_code'].astype(str)

    # Prepare features for prediction
    features = new_data[['county', 'state', 'year', 'month', 'total_customers',
                         'min_pct_covered', 'max_pct_covered', 'Customers']].values
    return model.predict(features)

# Example usage of the predict_outages function
# new_outage_data = pd.DataFrame({...})  # DataFrame containing new data
# predictions = predict_outages(new_outage_data)
# print(predictions)
