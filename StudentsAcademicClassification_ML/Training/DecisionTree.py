import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.preprocessing import LabelEncoder


# Automatically avoid stratify crash if class has < 2 instances
def safe_split(X, y, train_size=0.8, random_state=42):
    if all(v >= 2 for v in Counter(y).values()):
        return train_test_split(X, y, train_size=train_size, stratify=y, random_state=random_state)
    else:
        print("Warning: stratify skipped due to low sample size in some classes.")
        return train_test_split(X, y, train_size=train_size, random_state=random_state)


# Train and evaluate a decision tree model
def train_decision_tree_model(file_path, train_split, criterion, max_depth):
    data = pd.read_csv(file_path)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Encode y if not integer
    if y.dtype != 'int' and y.dtype != 'int64':
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = safe_split(X, y, train_size=train_split, random_state=42)

    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


# Handle dataset path options
def decision_tree_model(dataset="original", normalization="z_score", train_split=0.8, criterion="gini", max_depth=None):
    file_mapping = {
        "feature_engineered": f"../Normalized_Datasets/Train/train_{normalization}.csv",
        "original": f"../Normalized_Datasets/Train/raw_{normalization}.csv",
        "original_no_normalization": f"../Raw_Datasets/train.csv"
    }

    if dataset in file_mapping:
        file_path = file_mapping[dataset]
        return train_decision_tree_model(file_path, train_split, criterion, max_depth)
    else:
        raise ValueError("Dataset key not recognized!")


# Try all combinations
def run_decision_tree_combinations(datasets, criteria, max_depths, normalizations=None):
    for dataset in datasets:
        for criterion in criteria:
            for max_depth in max_depths:
                if normalizations:
                    for normalization in normalizations:
                        print(f"\n{dataset} | norm = {normalization} | criterion = {criterion} | depth = {max_depth}")
                        decision_tree_model(dataset=dataset, normalization=normalization,
                                            criterion=criterion, max_depth=max_depth)
                else:
                    print(f"\n{dataset} | criterion = {criterion} | depth = {max_depth}")
                    decision_tree_model(dataset=dataset, criterion=criterion, max_depth=max_depth)


# Run all configurations
if __name__ == "__main__":
    datasets = ["original", "feature_engineered"]
    dataset_raw = ["original_no_normalization"]
    normalizations = ["decimal_scaled", "z_score_scaled", "min_max_scaled"]
    criteria = ["gini", "entropy"]
    max_depths = [5, 15, 20, 25]

    run_decision_tree_combinations(datasets, criteria, max_depths, normalizations)
    run_decision_tree_combinations(dataset_raw, criteria, max_depths)
