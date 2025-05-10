import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def clean_column_names(df):
    df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
    return df


def train_lgbm_model(file_path, train_split, num_leaves, n_estimators, max_depth):
    data = pd.read_csv(file_path, sep=';')
    data = clean_column_names(data)
    if 'Target' not in data.columns:
        raise Exception(f"'Target' column not found in file: {file_path}")

    X = data.drop(columns=['Target'])
    y = data['Target']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=(1 - train_split), random_state=42)

    clf = lgb.LGBMClassifier(
        num_leaves=num_leaves,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        verbose=-1
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def lgbm_model(dataset="original", normalization="raw", train_split=0.8, num_leaves=31, n_estimators=100, max_depth=-1):
    file_mapping = {
        "feature_engineered": f"../Normalized_Datasets/Train/train_{normalization}_scaled.csv",
        "original": f"../Normalized_Datasets/Train/raw_{normalization}_scaled.csv",
        "original_no_normalization": f"../Raw_Datasets/train.csv"
    }

    if dataset in file_mapping:
        return train_lgbm_model(file_mapping[dataset], train_split, num_leaves, n_estimators, max_depth)
    else:
        raise Exception("No such dataset!")


def run_lgbm_combinations(datasets, normalizations, num_leaves_list, n_estimators_list, max_depth_list):
    results = []
    for dataset in datasets:
        for norm in normalizations:
            for num_leaves in num_leaves_list:
                for n_est in n_estimators_list:
                    for max_d in max_depth_list:
                        print(f"Running on {dataset} with {norm} normalization")
                        print(f"num_leaves={num_leaves}, n_estimators={n_est}, max_depth={max_d}")
                        accuracy = lgbm_model(
                            dataset=dataset,
                            normalization=norm,
                            num_leaves=num_leaves,
                            n_estimators=n_est,
                            max_depth=max_d
                        )
                        results.append({
                            "dataset": dataset,
                            "normalization": norm,
                            "num_leaves": num_leaves,
                            "n_estimators": n_est,
                            "max_depth": max_d,
                            "accuracy": accuracy
                        })
                        print(f"→ Accuracy: {accuracy:.4f}")
                        print("---")

    print("\n=== SUMMARY OF ALL RESULTS ===")
    for res in results:
        print(f"{res['dataset']} | {res['normalization']} | num_leaves={res['num_leaves']}, "
              f"n_estimators={res['n_estimators']}, max_depth={res['max_depth']} → "
              f"Accuracy: {res['accuracy']:.4f}")


if __name__ == "__main__":
    datasets = ["original", "feature_engineered"]
    normalizations = ["decimal", "z_score", "min_max"]
    num_leaves_list = [31]
    n_estimators_list = [50, 100, 150]
    max_depth_list = [-1, 5, 10]

    run_lgbm_combinations(datasets, normalizations, num_leaves_list, n_estimators_list, max_depth_list)
