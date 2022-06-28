import json

sklearn_ml_algo_sections = [".linear_model", ".cluster", ".compose", ".covariance", 
".cross_decomposition", ".decomposition", ".discriminant_analysis",
".ensemble", ".gaussian_process", ".isotonic", ".multiclass", ".multioutput",
".naive_bayes", ".neighbors", ".neural_network", ".semi_supervised", ".svm", ".tree"
]

skelarn_hpo_sections = ["GridSearchCV", "HalvingGridSearchCV", "ParameterGrid", "ParameterSampler", "RandomizedSearchCV", "HalvingRandomizedSearchCV"]





def main(source_file, target_file, section, names=None):
    with open(source_file, "r", encoding="utf-8") as src:
        data = json.load(src)

    ml_algos = []

    for item in data:
        if any(x in item["full_name"] for x in section):
            if item["name"][0].isupper():
                if names:
                    if item["name"] in names:
                        ml_algos.append(item)
                else:
                    ml_algos.append(item)


    with open(target_file, "w+", encoding="utf-8") as outfile:
        json.dump(ml_algos, outfile, indent=4)


if __name__ == "__main__":
    main("sklearn_default_values.json", "sklearn_hpo.json")