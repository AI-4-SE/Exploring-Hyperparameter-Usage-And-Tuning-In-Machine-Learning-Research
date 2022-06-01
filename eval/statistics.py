import glob
import shutil
import json
import csv
from numpy import full
import pandas as pd

from collections import Counter


statistics_dir = "../src/statistics/sklearn/params/"
statistics_processed = "data/statistics_processed/"
sklearn_data = "data/sklearn/modules/sklearn_modules.json"

all_modules = []


def copy_statistic_files(target):
    for repo in glob.glob("results/*"):
        for csv_file in glob.glob(f"{repo}/statistics/*"):
            if csv_file.endswith("params.json"):
                name = csv_file.split("/")[-1]
                print(csv_file)
                print(target+name)
                shutil.copyfile(csv_file, target + name)


def find_module(name):
    with open(sklearn_data, "r") as source:
        data = json.load(source)
        try:
            module = next(filter(lambda x: name in x["name"], data))
            return module
        except StopIteration:
            return None


def count_classes():
    for name in glob.glob(f"{statistics_unprocessed}/*"):
        df = pd.read_csv(name)
        repo_modules = set()


        for elem in df["node"]:
            parts = elem.split("::::")
            artifact = parts[1]
            module_name = parts[2]
            option_values = "::::".join(parts[3:])

            if artifact.endswith(".py"):
                module = find_module(module_name)
                if module:
                    repo_modules.add(module["name"])
                    #print(elem)
                    #print("artifact: ", artifact)    
                    #print("module: ", module_name)
                    #print("option_values: ", option_values)

        for x in repo_modules:
            all_modules.append(x)

    print(Counter(all_modules))


def clean_statistics():
    for full_name in glob.glob(f"{statistics_unprocessed}/*"):
        name = full_name.split("/")[-1]
        with open(f"{statistics_processed}{name}", "w+") as dest_file:
            with open(full_name, "r") as src_file:
                for line in src_file.readlines():
                    if "node" in line:
                        dest_file.write(line)
                    if ".py" in line:
                        dest_file.write(line)
        
        
def main():
    # count_classes()
    # clean_statistics()
    copy_statistic_files(statistics_dir)


if __name__ == "__main__":
    main()