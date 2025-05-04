import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Load training and testing datasets
raw_train = pd.read_csv("../Raw_Datasets/train.csv")
raw_test = pd.read_csv("../Raw_Datasets/test.csv")

# Define categorical and target column
categorical_features = [
    'Gender', 'Debtor', 'Scholarship holder', 'Displaced',
    'Educational special needs', 'Tuition fees up to date',
    'International', 'Marital status', 'Application order',
    'Daytime/evening attendance\t'
]
target_column = 'Target'

# Encode target column as integer classes
le = LabelEncoder()
raw_train[target_column] = le.fit_transform(raw_train[target_column])
raw_test[target_column] = le.transform(raw_test[target_column])

# Define normalization functions
def decimal_scaling(train_data, test_data, numerical_features):
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()
    for feature in numerical_features:
        magnitude = 10 ** (np.ceil(np.log10(np.abs(train_data[feature]).max())))
        train_scaled[feature] = train_data[feature] / magnitude
        if feature in test_data.columns:
            test_scaled[feature] = test_data[feature] / magnitude
    return train_scaled, test_scaled

def min_max_normalizer(train_data, test_data, numerical_features):
    min_max_scaler = MinMaxScaler()
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()
    train_scaled[numerical_features] = min_max_scaler.fit_transform(train_data[numerical_features])
    test_scaled[numerical_features] = min_max_scaler.transform(test_data[numerical_features])
    return train_scaled, test_scaled

def z_score_normalizer(train_data, test_data, numerical_features):
    standard_scaler = StandardScaler()
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()
    train_scaled[numerical_features] = standard_scaler.fit_transform(train_data[numerical_features])
    test_scaled[numerical_features] = standard_scaler.transform(test_data[numerical_features])
    return train_scaled, test_scaled

# Identify numerical columns for raw normalization
raw_train_features = [col for col in raw_train.columns if col not in categorical_features + [target_column]]
raw_test_features = [col for col in raw_test.columns if col not in categorical_features + [target_column]]

# Normalize raw datasets using all 3 methods
raw_normalized_datasets = {}

raw_train_dec, raw_test_dec = decimal_scaling(raw_train, raw_test, raw_train_features)
raw_normalized_datasets['decimal_scaled'] = (raw_train_dec, raw_test_dec)

raw_train_mm, raw_test_mm = min_max_normalizer(raw_train, raw_test, raw_train_features)
raw_normalized_datasets['min_max_scaled'] = (raw_train_mm, raw_test_mm)

raw_train_z, raw_test_z = z_score_normalizer(raw_train, raw_test, raw_train_features)
raw_normalized_datasets['z_score_scaled'] = (raw_train_z, raw_test_z)

# Save normalized raw datasets
for norm_name, (train_set, test_set) in raw_normalized_datasets.items():
    train_set.to_csv(f"../Normalized_Datasets/Train/raw_{norm_name}.csv", index=False)
    test_set.to_csv(f"../Normalized_Datasets/Test/raw_{norm_name}.csv", index=False)

print("Raw normalized datasets saved.")

# Define feature engineering function
def feature_engineering(data):
    data = data.copy()
    epsilon = 1e-9
    data['total_enrolled'] = data['Curricular units 1st sem (enrolled)'] + data['Curricular units 2nd sem (enrolled)']
    data['total_approved'] = data['Curricular units 1st sem (approved)'] + data['Curricular units 2nd sem (approved)']
    data['avg_grade'] = (data['Curricular units 1st sem (grade)'] + data['Curricular units 2nd sem (grade)']) / 2
    data['approval_ratio'] = (
        data['Curricular units 1st sem (approved)'] + data['Curricular units 2nd sem (approved)']
    ) / (
        data['Curricular units 1st sem (evaluations)'] + data['Curricular units 2nd sem (evaluations)'] + epsilon
    )
    return data

# Apply feature engineering to both datasets
train = feature_engineering(raw_train)
test = feature_engineering(raw_test)

# Re-encode Target just in case any transformation affected it
train[target_column] = le.transform(train[target_column])
test[target_column] = le.transform(test[target_column])

# Identify numerical columns after feature engineering
train_features = [col for col in train.columns if col not in categorical_features + [target_column]]
test_features = [col for col in test.columns if col not in categorical_features + [target_column]]

# Normalize feature-engineered datasets using all 3 methods
normalized_datasets = {}

train_dec, test_dec = decimal_scaling(train, test, train_features)
normalized_datasets['decimal_scaled'] = (train_dec, test_dec)

train_mm, test_mm = min_max_normalizer(train, test, train_features)
normalized_datasets['min_max_scaled'] = (train_mm, test_mm)

train_z, test_z = z_score_normalizer(train, test, train_features)
normalized_datasets['z_score_scaled'] = (train_z, test_z)

# Save normalized feature-engineered datasets
for norm_name, (train_set, test_set) in normalized_datasets.items():
    train_set.to_csv(f"../Normalized_Datasets/Train/train_{norm_name}.csv", index=False)
    test_set.to_csv(f"../Normalized_Datasets/Test/test_{norm_name}.csv", index=False)

print("Feature-engineered normalized datasets saved.")
