import pandas as pd
from sklearn.metrics import cohen_kappa_score


def agreement_research_questions():
    df = pd.read_csv("../../data/paper_analysis/annotator-agreement.csv")

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

    print("Score Question Model Parameter: ", round(model_parameter_score, 2))
    print("Score Question Final Values: ", round(final_values_score, 2))
    print("Score Question Hyperparameter: ", round(hyperparameter_score, 2))
    print("Score Question Technique: ", round(technique_score, 2))


def agreement_domains():
    df = pd.read_csv("../../data/paper_analysis/cross-validation-domains_new.csv")

    domains_rating1 = df["annotator 1"].tolist()
    domains_rating2 = df["annotator 2"].tolist()

    domains_score = cohen_kappa_score(domains_rating1, domains_rating2)

    print("Score Domains: ", round(domains_score, 2))


if __name__ == "__main__":
    agreement_research_questions()
    agreement_domains()