import glob
import json
import re
from xmlrpc.client import boolean
import plotly.graph_objs as go
import pandas as pd
from typing import List, Dict, Tuple
from ml_object import SklearnModule


STATISTICS_DIR = "data/statistics/"
SKL_MODULES_FILE = "data/sklearn/modules/sklearn_modules.json"
ALL_PROJECTS = "results/sklearn/statistics/*"
DEFAULT_REGEX = re.compile(r".+=.+")


def option_exists(option: List, params: List) -> boolean:
    param_names = [param[0] for param in params]
    if option[0] in param_names:
        return True
    else:
        return False


def get_key_value_pairs(data: Dict) -> List[Tuple]:
    return [(key, value) for key, value in data.items()]


def check_for_default_values(module_name: str, option_data: List[Tuple], sklearn_data: Dict) -> List[Tuple]:
    options = []

    for key in sklearn_data:
        if key["name"] == module_name:
            params = key["params"]
            for option in option_data:
                for param in params:
                    if option[0] == param[0]:
                        default = param[1]
                        if DEFAULT_REGEX.search(default):
                            default_parts = default.split("=")
                            default_value = default_parts[1]
                            if default_value == option[1]:
                                options.append((option[0], option[1], "default"))
                            else:
                                options.append((option[0], option[1], "custom"))
                        else:
                            options.append((option[0], option[1], "required"))
                    else: 
                        if option_exists(option, params):
                            continue
                        else:
                            options.append((option[0], option[1], "unknown"))

    return options


def create_modules(library_data) -> List:
    all_objects: List[SklearnModule] = []

    for project in glob.glob(ALL_PROJECTS):
        
        with open(project) as f:
            project_json = json.load(f)

            for key in project_json:
                file = project_json[key]
                if file:
                    for module in file:
                        sklearn_object_parts = module.split("_")
                        name = sklearn_object_parts[0]
                        line_nr = sklearn_object_parts[1]
                        option_data = get_key_value_pairs(file[module])
                        final_option_data = check_for_default_values(name, option_data, library_data)
                        all_objects.append(SklearnModule(name=name, file_name=file, line_nr=line_nr, options=final_option_data))

    return all_objects

def count_options(ml_modules) -> Dict:
    results = {}

    modules = {module.name for module in ml_modules}

    for module in modules:
        default_counter = 0
        custom_counter = 0
        required_counter = 0
        option_counter = 0
        unknown_counter = 0
        target_modules = list(filter(lambda x: x.name == module, ml_modules)) 
        for target_module in target_modules:
            for option in target_module.options:
                option_counter += 1
                if option[2] == "default":
                    default_counter += 1
                if option[2] == "custom":
                    custom_counter += 1
                if option[2] == "required":
                    required_counter += 1
                if option[2] == "unknown":
                    unknown_counter += 1

        results[module] = {
            "total": option_counter,
            "default": default_counter,
            "custom": custom_counter,
            "required": required_counter, 
            "unknown": unknown_counter,
        }
    
    return results


def get_modules_without_options(modules_option_count) -> List:
    modules_without_options = []

    for key in modules_option_count:
        module = modules_option_count[key]
        if module["total"] == 0:
            modules_without_options.append(key)

    return modules_without_options


def main():
    with open(SKL_MODULES_FILE) as f:
        sklearn_data = json.load(f)

    
    sklearn_modules = create_modules(sklearn_data)
    # tenforflow_modules = create_modules(tensorflow_data)
    # pytorch_modules = create_modules(pytorch_data)

    sklearn_option_count = count_options(sklearn_modules)
    # tensorflow_option_count = count_options(tensorflow_modules)
    # pytorch_option_count = count_options(pytorch_modules)

    with open("data/sklearn_option_count.json", "w") as f:
        json.dump(sklearn_option_count, f, sort_keys=True, indent=4)

    
    sklearn_modules_without_options = get_modules_without_options(sklearn_option_count)

    print(len(sklearn_modules_without_options))
    print(sklearn_modules_without_options)


if __name__ == "__main__":
    main()