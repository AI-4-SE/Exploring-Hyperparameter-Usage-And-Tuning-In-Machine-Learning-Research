import pandas as pd
from sklearn.metrics import cohen_kappa_score


def main():
    df = pd.read_csv("annotator-agreement.csv")

    model_parameter_rating1 = df["model parameter"].tolist()
    model_parameter_rating2 = df["model parameter 2"].tolist()

    final_values_rating1 = df["final values"].fillna("no").replace("not all", "yes").tolist()
    final_values_rating2 = df["final values 2"].tolist()

    hyperparameter_rating1 = df["hyperparameter"].tolist()
    hyperparameter_rating2 = df["hyperparameter 2"].tolist()

    technique_rating1 = df["technique"].fillna("nothing").tolist()
    technique_rating2 = df["technique 2"].fillna("nothing").tolist()

    model_parameter_score = cohen_kappa_score(model_parameter_rating1, model_parameter_rating2)
    final_values_score = cohen_kappa_score(final_values_rating1, final_values_rating2)
    hyperparameter_score = cohen_kappa_score(hyperparameter_rating1, hyperparameter_rating2)
    technique_score = cohen_kappa_score(technique_rating1, technique_rating2)

    print("Score Question Model Parameter: ", model_parameter_score)
    print("Score Question Final Values: ", final_values_score)
    print("Score Question Hyperparameter: ", hyperparameter_score)
    print("Score Question Technique: ", technique_score)


if __name__ == "__main__":
    main()