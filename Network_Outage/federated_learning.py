import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import tensorflow as tf

# Load datasets
outage_data = pd.read_csv('eaglei_outages_2014.csv')
coverage_data = pd.read_csv('coverage_history.csv')
modeled_customers_data = pd.read_csv('MCC.csv')
dqi_data = pd.read_csv('DQI.csv')

def process_and_train(outage_data, coverage_data, modeled_customers_data, dqi_data):

    # Data Preprocessing
    outage_data['run_start_time'] = pd.to_datetime(outage_data['run_start_time'])
    outage_data['year'] = outage_data['run_start_time'].dt.year.astype(int)
    outage_data['month'] = outage_data['run_start_time'].dt.month

    coverage_data['year'] = pd.to_datetime(coverage_data['year'], format='%m/%d/%y').dt.year

    state_abbreviations = {
        'Alabama': 'AL',
        'Alaska': 'AK',
        # Add other states as needed
    }
    outage_data['state'] = outage_data['state'].replace(state_abbreviations)
    outage_data['state'] = outage_data['state'].astype(str)
    coverage_data['state'] = coverage_data['state'].astype(str)
    dqi_data['fema'] = dqi_data['fema'].astype(str)

    outage_data['fips_code'] = outage_data['fips_code'].astype(str)
    modeled_customers_data['County_FIPS'] = modeled_customers_data['County_FIPS'].astype(str)

    label_encoder = LabelEncoder()
    outage_data['county'] = label_encoder.fit_transform(outage_data['county'])

    # Merging datasets
    data = outage_data.merge(coverage_data, on=['year', 'state'], how='left')
    data = data.merge(modeled_customers_data, left_on='fips_code', right_on='County_FIPS', how='left')
    data = data.merge(dqi_data, left_on=['year', 'state'], right_on=['year', 'fema'], how='left')

    data = data.rename(columns={'total_customers_x': 'total_customers'})
    data.drop(columns=['total_customers_y'], inplace=True)

    if 'county' in data.columns and 'state' in data.columns:
        data = pd.get_dummies(data, columns=['county', 'state'], drop_first=True)

    data.fillna(0, inplace=True)

    # Define features and target variable
    X = data.drop(columns=['customers_out', 'run_start_time', 'fips_code']).values
    y = data['customers_out'].values

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    return data, model, label_encoder




# Split data for two clients
def split_data_for_clients(data, client_count=2):
    data = shuffle(data, random_state=42)
    split_size = len(data) // client_count
    return [data.iloc[i * split_size:(i + 1) * split_size] for i in range(client_count)]

# Train a model for a single client
def train_client_model(data):
    X = data.drop(columns=['customers_out', 'run_start_time', 'fips_code']).values
    y = data['customers_out'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Client Model - Mean Squared Error: {mse}')
    print(f'Client Model - R^2 Score: {r2}')

    return model

# Aggregate model weights
def aggregate_model_weights(models):
    aggregated_weights = np.mean([model.feature_importances_ for model in models], axis=0)
    return aggregated_weights


# Preprocess and merge data
data, model, label_encoder = process_and_train(outage_data, coverage_data, modeled_customers_data, dqi_data)

# Split data for two clients
client_data = split_data_for_clients(data)

# Train models for each client
client_models = [train_client_model(client) for client in client_data]

# Aggregate model weights
aggregated_weights = aggregate_model_weights(client_models)
print("Aggregated Model Weights:", aggregated_weights)

# Calculate and print accuracies for each client model
def calculate_accuracies(client_models, client_data):
    for i, (model, data) in enumerate(zip(client_models, client_data)):
        X = data.drop(columns=['customers_out', 'run_start_time', 'fips_code']).values
        y = data['customers_out'].values
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        print(f"Client {i + 1} R^2 Score: {r2}")

# Print aggregated weights
print("Aggregated Model Weights:", aggregated_weights)

# Print accuracies for each client
calculate_accuracies(client_models, client_data)


# Calculate and print custom accuracy for each client model
def calculate_custom_accuracy(client_models, client_data, tolerance=0.1):
    for i, (model, data) in enumerate(zip(client_models, client_data)):
        X = data.drop(columns=['customers_out', 'run_start_time', 'fips_code']).values
        y = data['customers_out'].values
        y_pred = model.predict(X)

        # Custom accuracy: percentage of predictions within the tolerance
        accuracy = np.mean(np.abs(y - y_pred) <= tolerance * y) * 100
        print(f"Client {i + 1} Accuracy: {accuracy:.2f}%")


# Print custom accuracies for each client
calculate_custom_accuracy(client_models, client_data)
