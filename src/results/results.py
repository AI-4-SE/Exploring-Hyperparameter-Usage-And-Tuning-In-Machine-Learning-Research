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
import plotly.graph_objects as go
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


def get_fields(): 
    df = pd.read_csv(file_path)

    categories = []

    for _, row in df.iterrows():
        category = str(row["categories"])
        categories.append(category.strip())

    counter = Counter(categories)
    
    count = sum([count for _, count in counter.most_common()])
    assert count == 2000

    return [key for key, _ in counter.most_common()]


def analyze_field(df_field: pd.DataFrame):

    hp_yes = np.sum(df_field["hyperparameter"] == "yes")
    hp_no = np.sum(df_field["hyperparameter"] == "no")

    fv_yes = np.sum(df_field["final values"] == "yes")
    fv_no = np.sum(df_field["final values"] == "no")
    fv_not_all = np.sum(df_field["final values"] == "not all")

    return {
        "count": hp_yes + hp_no,
        "count_per": round((hp_yes + hp_no)/2000, 4),
        "hp_yes": hp_yes,
        "hp_yes_per": round(hp_yes/(hp_yes + hp_no), 2),
        "hp_no": hp_no,
        "hp_no_per": round(hp_no/(hp_yes + hp_no), 2),
        "fv_yes": fv_yes + fv_not_all,
        "fv_no": fv_no
    }

def hyperparameter_tuning_per_field():
    fields = get_fields()

    counter = []
    data = []

    for field in fields:
        df = pd.read_csv(file_path)
        df["final values"] = df["final values"].fillna("no")
        df_field = df[df['categories'] == field]
        #print("Field and length: ", field, len(df_field))

        counter.append(len(df_field))
        field_stats = analyze_field(df_field=df_field)
        field_stats["name"] = field
        data.append(field_stats)
    
    assert sum(counter) == 2000
    
    #print("Sum: ", sum(counter)) 
    #print(data)

    return data


def plot_hyperparameter_tuning_per_field():
    field_data = hyperparameter_tuning_per_field()

    field_names = []
    hp_yes = []
    hp_no = []

    for item in field_data:
        for key, value in item.items():
            if key == "name":
                field_names.append(value)
            if key == "hp_yes":
                hp_yes.append(value)
            if key == "hp_no":
                hp_no.append(value)        

    fig = go.Figure(data=[
        go.Bar(name='Not Reported', x=field_names, y=hp_no),
        go.Bar(name='Reported', x=field_names, y=hp_yes), 
    ])

    # Change the bar mode
    fig.update_layout(
        autosize=False,
        title="Hyperparameter Tuning Reporting in different ML fields.",
        barmode='stack', 
        bargap=0,
        width=800, 
        height=800,
        legend=dict(
        x=0.785,
        y=0.975,
    )
    )

    fig.update_traces(width=0.5)
    fig.update_xaxes(title_text='ML Fields', tickangle=90)
    fig.update_yaxes(title_text='Number of Research Papers')
    fig.show()

    fig.write_image("domains.svg")

def hyperparameter_per_field_table():
    field_data = hyperparameter_tuning_per_field()

    df = pd.DataFrame.from_records(field_data)
    df = df[["name", "count", "count_per", "hp_yes", "hp_yes_per", "hp_no", "hp_no_per"]]
    print(df.head())

    print(df.to_latex(index=False))
    
def hyperparameter_tuning_techniques():
    hp_tuning_techniques = ["grid search", "random search", "hyperparameter search", "experimental tuning", "hyperparameter sweep"]


    df = pd.read_csv(file_path)

    df = df[df["hyperparameter"] == "yes"]

    data = []
    for _, row in df.iterrows():
        techniques = str(row[12]).split(",")
        for x in techniques:
            
            tmp = x.strip().lower()
            if tmp:

                data.append(tmp)

    counter_techniques = Counter(data)

    print("Length techniques: ", len(counter_techniques.most_common()))

    for x in counter_techniques.most_common():
        print(x)


def sample_set_domain_analysis():
    df = pd.read_csv(file_path)
    df_sample = df.sample(100)

    df_sample.to_csv("../data/paper_analysis/cross-validation-domains.csv")


if __name__ == "__main__":
    hyperparameter_tuning()
    final_values()
    #sample_set_domain_analysis()
    #get_fields()
    #hyperparameter_tuning_per_field()
    #plot_hyperparameter_tuning_per_field()
    #hyperparameter_per_field_table()
    hyperparameter_tuning_techniques()