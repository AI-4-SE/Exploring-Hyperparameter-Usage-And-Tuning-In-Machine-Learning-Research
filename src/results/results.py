"""
This script calculates the results for the second research question: Are hyperparameters tuned and if so by which method?

Specifically, the script answer the following sub-questions:

(1) How many research paper write about hyperparameter tuning/final values?
(2) What ML fields report hyperparameter tuning?
(3) What hyperparameter are tuned?
    -- scikit learn
    -- tensorflow and pytorch
(4) Which techniques are used to tune hyperparameter?
"""

import pandas as pd
import numpy as np
from collections import Counter

file_path = "../data/paper_analysis/metadata-with-categories-cleaned.csv"


def hyperparameter_tuning():
    df = pd.read_csv(file_path)

    hp_yes = np.sum(df["hyperparameter"] == "yes")
    hp_no = np.sum(df["hyperparameter"] == "no")

    print("Hyperparameter Tuning (Yes/No/Total): ", hp_yes, hp_no, hp_yes + hp_no)
    print("Hyperparameter Tuning (Yes%/No&): ", round(hp_yes/(hp_yes + hp_no), 2), round(hp_no/(hp_yes + hp_no), 2))


def final_values():
    df = pd.read_csv(file_path)

    df["final values"] = df["final values"].fillna("no")

    fv_yes = np.sum(df["final values"] == "yes")
    fv_no = np.sum(df["final values"] == "no")
    fv_not_all = np.sum(df["final values"] == "not all")

    print("Final Values (Yes/No/NotAll/Total): ", fv_yes, fv_no, fv_not_all, fv_yes + fv_no + fv_not_all)
    print("Final Values (Yes/No/Total): ", fv_yes + fv_not_all, fv_no, fv_yes + fv_no + fv_not_all)
    print("Final Values (Yes%/No%): ", round((fv_yes + fv_not_all)/(fv_yes + fv_no + fv_not_all), 2), round((fv_no)/(fv_yes + fv_no + fv_not_all), 2))


def hyperparameter_tuning_per_field():
    pass


def hyperparameter_tuning_techniques():
    df = pd.read_csv(file_path)

    techniques = df["techniques"]

    counter_techniques = Counter(techniques)
    print("Most common hyperparameter tuning technique: ", counter_techniques.most_common(5))


if __name__ == "__main__":
    hyperparameter_tuning()
    final_values()
    hyperparameter_tuning_techniques()