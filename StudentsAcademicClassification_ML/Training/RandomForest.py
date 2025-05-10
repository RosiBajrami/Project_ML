import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

target_column = 'Target'


def train_rf_model(file_path, train_split, n_estimators, max_depth):
    data = pd.read_csv(file_path, sep=';')
    x = data.drop(columns=[target_column])
    y = data[target_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_split), random_state=42)

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    return acc


def rf_model(dataset="raw", normalization="raw", train_split=0.8, n_estimators=100, max_depth=None):
    file_mapping = {
        "feature_engineered": f"../Normalized_Datasets/Train/train_{normalization}.csv",
        "original": f"../Normalized_Datasets/Train/raw_{normalization}.csv",
        "original_no_normalization": f"../Raw_Datasets/train.csv"
    }

    if dataset in file_mapping:
        return train_rf_model(file_mapping[dataset], train_split, n_estimators, max_depth)
    else:
        raise Exception("No such dataset!")


def run_rf_combinations(datasets, normalizations, n_estimators_list, max_depth_list):
    for dataset in datasets:
        if dataset == "original_no_normalization":
            print(f"\nRunning on {dataset}")
            for n in n_estimators_list:
                for d in max_depth_list:
                    print(f"n_estimators={n}, max_depth={d}")
                    rf_model(dataset=dataset, n_estimators=n, max_depth=d)
                    print("---")
        else:
            for norm in normalizations:
                print(f"\nRunning on {dataset} with {norm} normalization")
                for n in n_estimators_list:
                    for d in max_depth_list:
                        print(f"n_estimators={n}, max_depth={d}")
                        rf_model(dataset=dataset, normalization=norm, n_estimators=n, max_depth=d)
                        print("---")


if __name__ == "__main__":
    datasets = ["original", "feature_engineered"]
    dataset_raw = ["original_no_normalization"]
    normalizations = ["decimal_scaled", "z_score_scaled", "min_max_scaled"]
    n_estimators_list = [50, 100, 150]
    max_depth_list = [None, 5, 10]

    run_rf_combinations(datasets, normalizations, n_estimators_list, max_depth_list)
    run_rf_combinations(dataset_raw, [], n_estimators_list, max_depth_list)

