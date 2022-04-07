import glob
import json
import re
import plotly.graph_objs as go
from typing import List, Dict, Tuple
from ml_object import SklearnModule
from eval_algorithm_specific import check_for_default_values, get_key_value_pairs, create_modules, count_options
import pandas as pd

STATISTICS_DIR = "data/statistics/"
SKL_MODULES_FILE = "data/sklearn/modules/sklearn_modules.json"
ALL_PROJECTS = "statistics/sklearn/statistics/*"


def create_specific_module(project_data, data):
    ml_module: List[SklearnModule] = []

    for key in project_data:
        file = project_data[key]
        if file:
            for module in file:
                sklearn_object_parts = module.split("_")
                name = sklearn_object_parts[0]
                line_nr = sklearn_object_parts[1]
                option_data = get_key_value_pairs(file[module])
                final_option_data = check_for_default_values(name, option_data, data)

                ml_module.append(SklearnModule(name=name, file_name=file, line_nr=line_nr, options=final_option_data))

    return ml_module


def get_counter(modules):
    default_counter = 0
    custom_counter = 0
    option_counter = 0
    required_counter = 0

    for sklearn_obj in modules:
        for option in sklearn_obj.options:
            option_counter += 1
            if option[2] == "default":
                default_counter += 1
            if option[2] == "custom":
                custom_counter += 1
            if option[2] == "required":
                required_counter += 1

    return {
        "default": default_counter,
        "total": option_counter,
        "custom": custom_counter,
        "required": required_counter,
    }


def plot_bar_chart(counter):
    params = ["total", "default", "custom", "required"]
    values = [counter["total"], counter["default"], counter["custom"], counter["required"]]

    data = [go.Bar(
        x=params,
        y=values
    )]

    fig = go.Figure(data=data)
    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
    )
    fig.show()


def plot_pie_chart(counter):
    labels = ["total options", "default options", "custom options", "not specified in api"]
    values = [counter["total"], counter["default"], counter["custom"], counter["required"]]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='percent+value')])
    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
    )
    fig.show()

def main():
    with open(SKL_MODULES_FILE) as f:
        sklearn_data = json.load(f)
    
    sklearn_modules = create_modules(sklearn_data)

    counter = get_counter(sklearn_modules)

    plot_bar_chart(counter) 
    plot_pie_chart(counter)

if __name__ == "__main__":
    main()