import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

target_column = 'Target'


def train_svm_model(file_path, train_split, kernel, degree):
    data = pd.read_csv(file_path, sep=';')
    x = data.drop(columns=[target_column])
    y = data[target_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_split), random_state=42)

    clf = svm.SVC(kernel=kernel, degree=degree if kernel == 'poly' else 3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    return acc


def svm_model(dataset="raw", normalization="raw", train_split=0.8, kernel="rbf", degree=0):
    file_mapping = {
        "feature_engineered": f"../Normalized_Datasets/Train/train_{normalization}.csv",
        "original": f"../Normalized_Datasets/Train/raw_{normalization}.csv",
        "original_no_normalization": f"../Raw_Datasets/train.csv"
    }

    if dataset in file_mapping:
        return train_svm_model(file_mapping[dataset], train_split, kernel, degree)
    else:
        raise Exception("No such dataset!")


def run_svm_combinations(datasets, kernels, degrees, normalizations=None):
    for dataset in datasets:
        for kernel in kernels:
            for degree in degrees:
                if normalizations:
                    for norm in normalizations:
                        print(f"Running {dataset} with {norm}, kernel={kernel}, degree={degree}")
                        svm_model(dataset=dataset, normalization=norm, kernel=kernel, degree=degree)
                        print("---")
                else:
                    print(f"Running {dataset} with kernel={kernel}, degree={degree}")
                    svm_model(dataset=dataset, kernel=kernel, degree=degree)
                    print("---")


if __name__ == "__main__":
    datasets = ["original", "feature_engineered"]
    dataset_raw = ["original_no_normalization"]
    normalizations = ["decimal_scaled", "z_score_scaled", "min_max_scaled"]
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    degrees = [0, 2, 3, 4]  # 0 means default handling

    run_svm_combinations(datasets, kernels, degrees, normalizations)
    run_svm_combinations(dataset_raw, kernels, degrees)

    ''' Example Best Result (to fill in after runs):
        Dataset: 
        Normalization: 
        Kernel:
        Degree: 
        Accuracy: 
    '''
