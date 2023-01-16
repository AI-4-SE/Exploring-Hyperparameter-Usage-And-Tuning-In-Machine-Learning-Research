import json

from sklearn.utils import all_estimators

def print_data():
    estimators = all_estimators(type_filter=["regressor", "cluster", "classifier"])
    estimator_names = [x[0] for x in estimators]

    optimizer = ["GridSearchCV", "HalvingGridSearchCV", "RandomizedSearchCV", "HalvingRandomizedSearchCV"]
    
    data = estimator_names + optimizer

    for x in data:
        print(x)
    print(len(data))


def main():
    
    sklearn_estimators = []
    sklearn_exp_settings = []

    estimators = all_estimators(type_filter=["classifier", "regressor", "cluster"])
    estimator_names = [x[0] for x in estimators]

    with open("modules/sklearn_default_values.json", "r", encoding="utf-8") as src:
        data = json.load(src)

    for item in data:
        if item["name"] in estimator_names:
            sklearn_estimators.append(item)
        else:
            sklearn_exp_settings.append(item)
        
    
    with open("modules/sklearn_estimators.json", "w", encoding="utf-8") as src:
        json.dump(sklearn_estimators, src, indent=4, sort_keys=True)
    
    with open("modules/sklearn_exp_settings.json", "w", encoding="utf-8") as src:
        json.dump(sklearn_exp_settings, src, indent=4, sort_keys=True)


if __name__ == "__main__":
    print_data()
    main()
