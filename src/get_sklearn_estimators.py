import json

from sklearn.utils import all_estimators

def print_estimators():
    estimators = all_estimators(type_filter="transformer")
    estimator_names = [x[0] for x in estimators]
    print(estimator_names)
    print(len(estimator_names))


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
    #print_estimators()
    main()
