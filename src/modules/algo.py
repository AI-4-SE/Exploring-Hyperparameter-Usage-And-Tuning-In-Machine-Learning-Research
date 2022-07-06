import json

sklearn_ml_algo_sections = [
    ".linear_model", ".cluster", ".compose", ".covariance", 
    ".cross_decomposition", ".decomposition", ".discriminant_analysis", ".kernel_ridge",
    ".ensemble", ".gaussian_process", ".isotonic", ".multiclass", ".multioutput",
    ".naive_bayes", ".neighbors", ".neural_network", ".semi_supervised", ".svm", ".tree", 
    ".mixture", ".pipeline"
]

sklearn_hpo_sections = [
    "GridSearchCV", "HalvingGridSearchCV", "ParameterGrid", 
    "ParameterSampler", "RandomizedSearchCV", "HalvingRandomizedSearchCV"
]


sklearn_exp_settings = [
    ".datasets", ".exceptions", ".feature_extraction", ".feature_selection", 
    ".impute", ".inspection", ".kernel_approximation", ".manifold", ".metrics"
    ".model_selection", ".preprocessing", ".random_projection", ".utils"
]


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


def experimental_settings(source_file, target_file, section):
    with open(source_file, "r", encoding="utf-8") as src:
        data = json.load(src)

    exp_settings = []

    for item in data:
        if any(x in item["full_name"] for x in section):
            if item["name"][0].isupper():
                exp_settings.append(item)


    with open(target_file, "w+", encoding="utf-8") as outfile:
        json.dump(exp_settings, outfile, indent=4)



if __name__ == "__main__":
    experimental_settings("sklearn_default_values.json", "sklearn_experimental_settings.json", sklearn_exp_settings)
    main("sklearn_default_values.json", "sklearn_ml_algorithms.json", sklearn_ml_algo_sections)