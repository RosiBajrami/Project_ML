import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the dataset
train = pd.read_csv("../Raw_Datasets/train.csv", sep=';')
test = pd.read_csv("../Raw_Datasets/test.csv", sep=';')

# Define categorical features and target column
categorical_features = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance\t', 'Previous qualification', 'Nacionality',
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
]
target_column = 'Target'

# Identify raw numerical features (exclude categorical + target, ensure numeric type)
raw_numerical_features = [
    col for col in train.columns
    if col not in categorical_features + [target_column]
    and pd.api.types.is_numeric_dtype(train[col])
]

# ---------- Normalization Functions ----------

def decimal_scaling(train_data, test_data, numerical_features):
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()
    for feature in numerical_features:
        magnitude = 10 ** (np.ceil(np.log10(np.abs(train_data[feature]).max())))
        if magnitude == 0:
            magnitude = 1
        train_scaled[feature] = train_data[feature] / magnitude
        test_scaled[feature] = test_data[feature] / magnitude
    return train_scaled, test_scaled

def min_max_normalizer(train_data, test_data, numerical_features):
    scaler = MinMaxScaler()
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()
    train_scaled[numerical_features] = scaler.fit_transform(train_data[numerical_features])
    test_scaled[numerical_features] = scaler.transform(test_data[numerical_features])
    return train_scaled, test_scaled

def z_score_normalizer(train_data, test_data, numerical_features):
    scaler = StandardScaler()
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()
    train_scaled[numerical_features] = scaler.fit_transform(train_data[numerical_features])
    test_scaled[numerical_features] = scaler.transform(test_data[numerical_features])
    return train_scaled, test_scaled

# ---------- Feature Engineering Function ----------

def feature_engineering(data, is_test=False):
    data = data.copy()
    data['total_units_enrolled'] = data['Curricular units 1st sem (enrolled)'] + data['Curricular units 2nd sem (enrolled)']
    data['total_units_approved'] = data['Curricular units 1st sem (approved)'] + data['Curricular units 2nd sem (approved)']
    data['total_evaluations'] = data['Curricular units 1st sem (evaluations)'] + data['Curricular units 2nd sem (evaluations)']
    total_grade = data['Curricular units 1st sem (grade)'] + data['Curricular units 2nd sem (grade)']
    data['approval_ratio'] = np.where(data['total_units_enrolled'] != 0, data['total_units_approved'] / data['total_units_enrolled'], 0)
    data['avg_grade_per_eval'] = np.where(data['total_evaluations'] != 0, total_grade / data['total_evaluations'], 0)

    # Drop original columns after combining
    columns_to_drop = [
        'Curricular units 1st sem (enrolled)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 2nd sem (evaluations)',
        'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)'
    ]
    data = data.drop(columns=columns_to_drop)

    return data, columns_to_drop

# ---------- Apply Raw Normalization ----------

raw_normalized_datasets = {}

raw_train_decimal_scaled, raw_test_decimal_scaled = decimal_scaling(train, test, raw_numerical_features)
raw_normalized_datasets['decimal_scaled'] = (raw_train_decimal_scaled, raw_test_decimal_scaled)

raw_train_min_max_scaled, raw_test_min_max_scaled = min_max_normalizer(train, test, raw_numerical_features)
raw_normalized_datasets['min_max_scaled'] = (raw_train_min_max_scaled, raw_test_min_max_scaled)

raw_train_z_score_scaled, raw_test_z_score_scaled = z_score_normalizer(train, test, raw_numerical_features)
raw_normalized_datasets['z_score_scaled'] = (raw_train_z_score_scaled, raw_test_z_score_scaled)

# Save raw normalized datasets
raw_output_path = "../Normalized_Datasets"
for norm_name, (raw_train_set, raw_test_set) in raw_normalized_datasets.items():
    raw_train_file_name = f"{raw_output_path}/Train/raw_{norm_name}.csv"
    raw_test_file_name = f"{raw_output_path}/Test/raw_{norm_name}.csv"
    raw_train_set.to_csv(raw_train_file_name, sep=';', index=False)
    raw_test_set.to_csv(raw_test_file_name, sep=';', index=False)

# ---------- Apply Feature Engineering ----------

train, dropped_columns = feature_engineering(train, is_test=False)
test, _ = feature_engineering(test, is_test=True)

# Update numerical features after dropping + adding engineered columns
numerical_features = [
    col for col in raw_numerical_features if col not in dropped_columns
] + ['total_units_enrolled', 'total_units_approved', 'total_evaluations', 'approval_ratio', 'avg_grade_per_eval']

# ---------- Apply Feature-Engineered Normalization ----------

normalized_datasets = {}

train_decimal_scaled, test_decimal_scaled = decimal_scaling(train, test, numerical_features)
normalized_datasets['decimal_scaled'] = (train_decimal_scaled, test_decimal_scaled)

train_min_max_scaled, test_min_max_scaled = min_max_normalizer(train, test, numerical_features)
normalized_datasets['min_max_scaled'] = (train_min_max_scaled, test_min_max_scaled)

train_z_score_scaled, test_z_score_scaled = z_score_normalizer(train, test, numerical_features)
normalized_datasets['z_score_scaled'] = (train_z_score_scaled, test_z_score_scaled)

# Save feature-engineered normalized datasets
output_path = "../Normalized_Datasets"
for norm_name, (train_set, test_set) in normalized_datasets.items():
    train_file_name = f"{output_path}/Train/train_{norm_name}.csv"
    test_file_name = f"{output_path}/Test/test_{norm_name}.csv"
    train_set.to_csv(train_file_name, sep=';', index=False)
    test_set.to_csv(test_file_name, sep=';', index=False)

# ---------- Final Summary ----------

print("Normalization completed. Files saved:")
print("Raw Normalized Datasets:")
for norm_name in raw_normalized_datasets.keys():
    print(f"- raw_train_{norm_name}.csv")
    print(f"- raw_test_{norm_name}.csv")
print("\nFeature-Engineered Normalized Datasets:")
for norm_name in normalized_datasets.keys():
    print(f"- train_{norm_name}.csv")
    print(f"- test_{norm_name}.csv")
