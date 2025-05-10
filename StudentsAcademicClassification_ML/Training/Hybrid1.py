import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(file_path, target_column='Target'):
    data = pd.read_csv(file_path, sep=';')
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in {file_path}")
    x = data.drop(columns=[target_column])
    y = data[target_column]
    return x, y

def add_kmeans_clusters(x, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(x)
    x_with_clusters = x.copy()
    x_with_clusters['ClusterLabel'] = cluster_labels
    return x_with_clusters

def train_rf_with_clusters(x, y, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

if __name__ == "__main__":
    datasets = {
        "original": "../Normalized_Datasets/Train/raw",
        "feature_engineered": "../Normalized_Datasets/Train/train",
        "original_no_normalization": "../Raw_Datasets/train"
    }
    normalizations = ["decimal_scaled", "min_max_scaled", "z_score_scaled"]
    n_clusters = 10  # increased from 5 to 10
    n_estimators = 300  # increased from 100 to 300
    max_depth = 20  # limited tree depth to avoid overfitting
    min_samples_split = 2
    min_samples_leaf = 1

    for dataset_type, base_path in datasets.items():
        if dataset_type == "original_no_normalization":
            file_path = f"{base_path}.csv"
            print(f"\nRunning on {dataset_type} (no normalization)")
            try:
                x, y = load_data(file_path)
                x_with_clusters = add_kmeans_clusters(x, n_clusters)
                accuracy = train_rf_with_clusters(x_with_clusters, y, n_estimators, max_depth, min_samples_split, min_samples_leaf)
                print(f"Accuracy (RF + KMeans feature): {accuracy:.4f}")
            except Exception as e:
                print(f"Error with {file_path}: {e}")
        else:
            for norm in normalizations:
                file_path = f"{base_path}_{norm}.csv"
                print(f"\nRunning on {dataset_type} with {norm} normalization")
                try:
                    x, y = load_data(file_path)
                    x_with_clusters = add_kmeans_clusters(x, n_clusters)
                    accuracy = train_rf_with_clusters(x_with_clusters, y, n_estimators, max_depth, min_samples_split, min_samples_leaf)
                    print(f"Accuracy (RF + KMeans feature): {accuracy:.4f}")
                except Exception as e:
                    print(f"Error with {file_path}: {e}")
