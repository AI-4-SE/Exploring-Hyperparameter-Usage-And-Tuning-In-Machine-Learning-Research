import glob
import shutil
import json
import csv
import pandas as pd

from collections import Counter


statistics_unprocessed = "data/statistics_unprocessed/"
statistics_processed = "data/statistics_processed/"
sklearn_data = "data/sklearn/modules/sklearn_modules.json"

all_modules = []


def copy_statistic_files():
    for repo in glob.glob("results/*"):
        for csv_file in glob.glob(f"{repo}/statistics/*"):
            name = csv_file.split("/")[-1]
            print(csv_file)
            shutil.copyfile(csv_file, statistics_unprocessed + name)
            break


def find_module(name):
    with open(sklearn_data, "r") as source:
        data = json.load(source)
        try:
            module = next(filter(lambda x: name in x["name"], data))
            return module
        except StopIteration:
            return None


def clean_statistics():
    for name in glob.glob(f"{statistics_unprocessed}/*"):
        df = pd.read_csv(name)
        repo_modules = set()


        for elem in df["node"]:
            parts = elem.split("::::")
            artifact = parts[1]
            module_name = parts[2]
            option_values = "::::".join(parts[3:])

            if artifact.endswith(".py"):
                module = find_module(module_name)
                if module:
                    repo_modules.add(module["name"])
                    #print(elem)
                    #print("artifact: ", artifact)    
                    #print("module: ", module_name)
                    #print("option_values: ", option_values)

        for x in repo_modules:
            all_modules.append(x)

    print(Counter(all_modules))



def main():
    #clean_statistics()

    data = {'KMeans': 16, 'GroupKFold': 14, 'LogisticRegression': 12, 'MinMaxScaler': 8, 'LinearSVC': 8, 'TSNE': 7, 'LinearRegression': 7, 'IncrementalPCA': 6, 'StandardScaler': 6, 'GridSearchCV': 6, 'BayesianGaussianMixture': 5, 'RandomForestClassifier': 5, 'Pipeline': 5, 'TfidfVectorizer': 5, 'RepeatedStratifiedKFold': 5, 'MultiLabelBinarizer': 5, 'TruncatedSVD': 5, 'CountVectorizer': 5, 'DBSCAN': 4, 'LabelEncoder': 4, 'KDTree': 3, 'NearestNeighbors': 3, 'DecisionTreeClassifier': 3, 'AgglomerativeClustering': 3, 'OneHotEncoder': 3, 'PolynomialFeatures': 3, 'KernelRidge': 3, 'RandomForestRegressor': 3, 'GroupShuffleSplit': 2, 'ColumnTransformer': 2, 'LatentDirichletAllocation': 2, 'FunctionTransformer': 2, 'DummyClassifier': 2, 'ParameterGrid': 2, 'KNeighborsClassifier': 2, 'GraphicalLasso': 2, 'TimeSeriesSplit': 2, 'DecisionTreeRegressor': 2, 'SGDClassifier': 1, 'SGDRegressor': 1, 'Perceptron': 1, 'SGDOneClassSVM': 1, 'MaxAbsScaler': 1, 'LeaveOneGroupOut': 1, 'MLPClassifier': 1, 'GaussianProcessClassifier': 1, 'OneVsRestClassifier': 1, 'RBF': 1, 'BernoulliNB': 1, 'TfidfTransformer': 1, 'MultinomialNB': 1, 'ComplementNB': 1, 'IsotonicRegression': 1, 'MiniBatchKMeans': 1, 'MeanShift': 1, 'KernelDensity': 1, 'RepeatedKFold': 1, 'DummyRegressor': 1, 'RegressorChain': 1, 'SimpleImputer': 1, 'MultiOutputRegressor': 1, 'RidgeCV': 1, 'ElasticNetCV': 1, 'RBFSampler': 1, 'GradientBoostingRegressor': 1, 'GaussianNB': 1, 'LinearSVR': 1, 'KernelPCA': 1, 'LabelBinarizer': 1, 'KBinsDiscretizer': 1, 'IterativeImputer': 1, 'GaussianProcessRegressor': 1, 'FeatureUnion': 1, 'KNeighborsRegressor': 1, 'MLPRegressor': 1, 'HuberRegressor': 1, 'TransformedTargetRegressor': 1, 'ParameterSampler': 1}

    print(data)

    with open("data/modules_all.txt", "w") as file:
        for x, y in zip(data.keys(), data.values()):
            file.write(f"{x}: {y}\n")

if __name__ == "__main__":
    main()