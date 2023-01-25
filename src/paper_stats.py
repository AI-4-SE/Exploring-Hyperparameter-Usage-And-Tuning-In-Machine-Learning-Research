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
from typing import Dict
import matplotlib.pyplot as plt
from aquarel import load_theme
import glob
import json

file_path = "../data/paper_analysis/metadata.csv"


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


def clean_dates():
    dates_cleaned = []

    df = pd.read_csv("../data/paper_analysis/metadata-with-dates.csv")
    dates = df["date"].tolist()

    dates_cleaned = [x.split("-")[0] for x in dates]

    df["dates cleaned"] = dates_cleaned
    
    df.to_csv("../data/paper_analysis/metadata-with-dates-cleaned.csv")


def get_date_values():
    df = pd.read_csv("../data/paper_analysis/metadata-with-dates-cleaned.csv")
    return sorted(df['dates cleaned'].unique().tolist())


def get_paper_per_year(year: str):
    df = pd.read_csv("../data/paper_analysis/metadata-with-dates-cleaned.csv")
    df["final values"] = df["final values"].fillna("no")
    df = df[df["dates cleaned"] == year]

    hp_yes = np.sum(df["hyperparameter"] == "yes")
    hp_no = np.sum(df["hyperparameter"] == "no")
    hp_total = hp_yes + hp_no

    fv_yes = np.sum(df["final values"] == "yes")
    fv_no = np.sum(df["final values"] == "no")
    fv_not_all = np.sum(df["final values"] == "not all")
    fv_yes_not_all = fv_yes + fv_not_all
    fv_total = fv_yes_not_all + fv_no

    hp_yes_per = int(round(hp_yes/hp_total, 1) * 100)
    hp_no_per = int(round(hp_no/hp_total, 1) * 100)
    fv_yes_per = int(round(fv_yes_not_all/fv_total, 1) * 100)
    fv_no_per = int(round(fv_no/fv_total, 1) * 100)
    

    assert hp_yes + hp_no == fv_yes + fv_not_all + fv_no
    assert hp_yes_per + hp_no_per == 100
    assert fv_yes_per + fv_no_per == 100

    return {
        "year": year,
        "count": hp_yes + hp_no,
        "count_per": round((hp_yes + hp_no)/2000, 4),
        "hp_yes": hp_yes,
        "hp_yes_per": hp_yes_per,
        "hp_no": hp_no,
        "hp_no_per": hp_no_per,
        "fv_yes": fv_yes + fv_not_all,
        "fv_yes_per": fv_yes_per,
        "fv_no": fv_no,
        "fv_no_per": fv_no_per,
    }

def create_paper_per_year_percentage():
    data = []

    dates = get_date_values()
    for date in dates[8:]:
        data.append(get_paper_per_year(date))

    labels = [x["year"] for x in data]
    hp_yes = [x["hp_yes_per"] for x in data]
    hp_no = [x["hp_no_per"] for x in data]
    fv_yes = [x["fv_yes_per"] for x in data]
    fv_no = [x["fv_no_per"] for x in data]

    hp_data = pd.DataFrame({
        "Report HP Tuning: Yes": hp_yes,
        "Report HP Tuning: No": hp_no,
        }, index=labels
    )

    fv_data = pd.DataFrame({
        "Report HP Values: Yes": fv_yes,
        "Report HP Values: No": fv_no,
        }, index=labels
    )
    
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the bar width and space between bars
    bar_width = 0.35
    bar_space = 0.05

    # Create the bars for hp_data
    hp_yes = ax.bar(np.arange(len(hp_data)) - bar_width/2 - bar_space, hp_data["Report HP Tuning: Yes"], bar_width, color = '#1f77b4', label="Report HP Tuning: Yes")
    hp_no = ax.bar(np.arange(len(hp_data)) - bar_width/2 - bar_space, hp_data["Report HP Tuning: No"], bar_width, bottom=hp_data["Report HP Tuning: Yes"], color = '#1f77b4', alpha=0.5, label="Report HP Tuning: No")

    # Create the bars for fv_data
    fv_yes = ax.bar(np.arange(len(fv_data)) + bar_width/2 + bar_space, fv_data["Report HP Values: Yes"], bar_width, color = '#009e73', label="Report HP Values: Yes")
    fv_no = ax.bar(np.arange(len(fv_data)) + bar_width/2 + bar_space, fv_data["Report HP Values: No"], bar_width, bottom=fv_data["Report HP Values: Yes"], color = '#009e73', alpha=0.5, label="Report HP Values: No")

    # Set the x-axis labels
    ax.set_xticks(np.arange(len(hp_data)))
    ax.set_xticklabels(hp_data.index)

    # Add a legend
    ax.legend()

    # Add a horizontal grid
    ax.grid(axis='y', linestyle='-', color='gray', alpha=0.5)
    ax.set_axisbelow(True)

    ax.set_ylabel('Percentage of Research Paper')
    ax.set_xlabel('Year')
    ax.set_title('Reporting Practices of Hyperparameter in Research Papers')

    # Show the plot
    #plt.show()
    plt.savefig("paper_per_year_percentage.png",bbox_inches='tight',  pad_inches=0)


def create_paper_per_year_absolute_numbers():
    data = []

    dates = get_date_values()
    for date in dates[8:]:
        data.append(get_paper_per_year(date))

    #for x in data:
    #    print(x)

    labels = [x["year"] for x in data]
    hp_yes = [x["hp_yes"] for x in data]
    hp_no = [x["hp_no"] for x in data]
    fv_yes = [x["fv_yes"] for x in data]
    fv_no = [x["fv_no"] for x in data]


    hp_data = pd.DataFrame({
        "Report HP Tuning: Yes": hp_yes,
        "Report HP Tuning: No": hp_no,
        }, index=labels
    )

    fv_data = pd.DataFrame({
        "Report HP Values: Yes": fv_yes,
        "Report HP Values: No": fv_no,
        }, index=labels
    )

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the bar width and space between bars
    bar_width = 0.35
    bar_space = 0.025

    # Create the bars for hp_data
    hp_yes = ax.bar(np.arange(len(hp_data)) - bar_width/2 - bar_space, hp_data["Report HP Tuning: Yes"], bar_width, color = '#1f77b4', label="Report HP Tuning: Yes")
    hp_no = ax.bar(np.arange(len(hp_data)) - bar_width/2 - bar_space, hp_data["Report HP Tuning: No"], bar_width, bottom=hp_data["Report HP Tuning: Yes"], color = '#1f77b4', alpha=0.5, label="Report HP Tuning: No")

    # Create the bars for fv_data
    fv_yes = ax.bar(np.arange(len(fv_data)) + bar_width/2 + bar_space, fv_data["Report HP Values: Yes"], bar_width, color = '#009e73', label="Report HP Values: Yes")
    fv_no = ax.bar(np.arange(len(fv_data)) + bar_width/2 + bar_space, fv_data["Report HP Values: No"], bar_width, bottom=fv_data["Report HP Values: Yes"], color = '#009e73', alpha=0.5, label="Report HP Values: No")

    # Set the x-axis labels
    ax.set_xticks(np.arange(len(hp_data)))
    ax.set_xticklabels(hp_data.index)

    # Add a legend
    ax.legend()

    # Add a horizontal grid
    ax.grid(axis='y', linestyle='-', color='gray', alpha=0.5)
    ax.set_axisbelow(True)

    # Set the y-limits
    y_max = max(hp_data.sum(axis=1).max(), fv_data.sum(axis=1).max())
    ax.set_ylim(0, y_max+100)

    # Add the line chart
    #ax.plot(np.arange(len(hp_data)), hp_data.sum(axis=1), '-o', color='#000000', label='Total HP')
    ax.plot(np.arange(len(fv_data)), fv_data.sum(axis=1), '-o', color='#000000')
    
    ax.set_ylabel('Number of Research Papers')
    ax.set_xlabel('Year')
    ax.set_title('Reporting Practices of Hyperparameter in Research Papers')

    # Show the plot
    #plt.show()
    plt.savefig("paper_per_year_absolute.png",bbox_inches='tight',  pad_inches=0)


def create_paper_per_year_absolute_numbers_and_dbl_count():
    # trigger core fonts for PDF backend
    plt.rcParams["pdf.use14corefonts"] = True

    data = []

    dates = get_date_values()
    for date in dates[8:]:
        data.append(get_paper_per_year(date))

    #for x in data:
    #    print(x)

    labels = [x["year"] for x in data]
    hp_yes = [x["hp_yes"] for x in data]
    hp_no = [x["hp_no"] for x in data]
    fv_yes = [x["fv_yes"] for x in data]
    fv_no = [x["fv_no"] for x in data]

    hp_data = pd.DataFrame({
        "Report HP Tuning: Yes": hp_yes,
        "Report HP Tuning: No": hp_no,
        }, index=labels
    )

    fv_data = pd.DataFrame({
        "Report HP Values: Yes": fv_yes,
        "Report HP Values: No": fv_no,
        }, index=labels
    )

    dblp_data = pd.DataFrame({
        "Count": [19, 19, 35, 45, 100, 124, 146],
        }, index=labels
    )

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the bar width and space between bars
    bar_width = 0.35
    bar_space = 0.025

    # Create the bars for hp_data
    hp_yes = ax.bar(np.arange(len(hp_data)) - bar_width/2 - bar_space, hp_data["Report HP Tuning: Yes"], bar_width, color = '#1f77b4', label="Report HP Tuning: Yes")
    hp_no = ax.bar(np.arange(len(hp_data)) - bar_width/2 - bar_space, hp_data["Report HP Tuning: No"], bar_width, bottom=hp_data["Report HP Tuning: Yes"], color = '#1f77b4', alpha=0.5, label="Report HP Tuning: No")

    # Create the bars for fv_data
    fv_yes = ax.bar(np.arange(len(fv_data)) + bar_width/2 + bar_space, fv_data["Report HP Values: Yes"], bar_width, color = '#009e73', label="Report HP Values: Yes")
    fv_no = ax.bar(np.arange(len(fv_data)) + bar_width/2 + bar_space, fv_data["Report HP Values: No"], bar_width, bottom=fv_data["Report HP Values: Yes"], color = '#009e73', alpha=0.5, label="Report HP Values: No")

    # Set the x-axis labels
    ax.set_xticks(np.arange(len(hp_data)))
    ax.set_xticklabels(hp_data.index)

    # Add a legend
    ax.legend()

    # Add a horizontal grid
    ax.grid(axis='y', linestyle='-', color='gray', alpha=0.5)
    ax.set_axisbelow(True)

    # Set the y-limits
    y_max = max(hp_data.sum(axis=1).max(), fv_data.sum(axis=1).max())
    ax.set_ylim(0, y_max+100)

    # Add the line chart
    ax2 = ax.twinx()
    ax2.set_ylim(0, 150)
    ax2.yaxis.set_tick_params(labelright=True)
    ax2.set_ylabel('DBLP Count')
    ax2.set_yticks(np.array([0, 25, 50, 75, 100, 125, 150, 175]))
    ax2.plot(np.arange(len(dblp_data)), dblp_data["Count"], '-o', color='#000000', label='DBLP Count')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.57, 1))
        
    ax.set_ylabel('Number of Research Papers')
    ax.set_xlabel('Year')
    ax.set_title('Reporting Practices of Hyperparameter in Research Papers')

    #plt.show()
    #plt.savefig("paper_per_year_absolute_and_dblp_count.png",bbox_inches='tight',  pad_inches=0)
    fig.savefig("paper_per_year_absolute_and_dblp_count.pdf", format="pdf", bbox_inches='tight',  pad_inches=0)



def get_statistic_files(df: pd.DataFrame):
    statistic_files = []

    repo_names = []
    files = glob.glob("../data/statistics/*")
    for x in files:
        repo_names.append(x.split("\\")[-1].split("_params.json")[0])
    
    urls = df["repo_url"].tolist()
    urls = [x.split("/")[-1] for x in urls]

    for url in urls:
        counter = 0
        for repo in repo_names:
            if repo == url:
                counter += 1
                statistic_files.append(repo + "_params.json")
                break
            
        if counter == 0:
            for repo in repo_names:
                if "_" + url in repo:
                    counter += 1
                    statistic_files.append(repo + "_params.json")
                    break

    return statistic_files


def get_statistic_files_per_year(year):
    df = pd.read_csv("../data/paper_analysis/metadata-with-dates-cleaned.csv")
    df["final values"] = df["final values"].fillna("no")
    df = df[df["dates cleaned"] == year]

    df_hp_yes = df[df["hyperparameter"] == "yes"]
    df_hp_no = df[df["hyperparameter"] == "no"]

    df_fv_yes = df[df["hyperparameter"] == "yes"]
    df_fv_no = df[df["hyperparameter"] == "no"]
    df_fv_not_all = df[df["hyperparameter"] == "not all"]
    df_yes_all = pd.concat([df_fv_yes, df_fv_not_all])

    urls_hp_yes = df_hp_yes["repo_url"].tolist()
    urls_hp_no = df_hp_no["repo_url"].tolist()
    urls_fv_yes = df_yes_all["repo_url"].tolist()
    urls_fv_no = df_fv_no["repo_url"].tolist()

    stats_files_hp_yes = get_statistic_files(df_hp_yes)
    stats_files_hp_no = get_statistic_files(df_hp_no)
    stats_files_fv_yes = get_statistic_files(df_yes_all)
    stats_files_fv_no = get_statistic_files(df_fv_no)

    assert len(urls_hp_yes) == len(stats_files_hp_yes)
    assert len(urls_hp_no) == len(stats_files_hp_no)
    assert len(urls_fv_yes) == len(stats_files_fv_yes)
    assert len(urls_fv_no) == len(stats_files_fv_no)

    return {
        "year": year,
        "hp_yes": stats_files_hp_yes,
        "hp_no": stats_files_hp_no,
        "fv_yes": stats_files_fv_yes,
        "fv_no": stats_files_fv_no
    }


def store_statistic_files_per_year():
    data = []
    dates = get_date_values()
    counter_hp = 0
    counter_fv = 0
    for date in dates:
        x = get_statistic_files_per_year(date)
        counter_hp += len(x["hp_yes"]) + len(x["hp_no"])
        counter_fv += len(x["fv_yes"]) + len(x["fv_no"])
        data.append(get_statistic_files_per_year(date))

    assert counter_hp == 2000
    assert counter_fv == 2000
        
    with open("../data/statistic_files_per_year.json", "w", encoding="utf-8") as dest:
        json.dump(data, dest, sort_keys=True, indent=4)

if __name__ == "__main__":
    #hyperparameter_tuning()
    #final_values()
    #sample_set_domain_analysis()
    #get_fields()
    #hyperparameter_tuning_per_field()
    #plot_hyperparameter_tuning_per_field()
    #hyperparameter_per_field_table()
    #hyperparameter_tuning_techniques()

    #create_paper_per_year_percentage()
    #create_paper_per_year_absolute_numbers()
    create_paper_per_year_absolute_numbers_and_dbl_count()

    #store_statistic_files_per_year()
