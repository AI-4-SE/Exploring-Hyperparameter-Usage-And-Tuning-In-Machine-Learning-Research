import glob
import shutil
import json
import csv
from numpy import full
import pandas as pd

from collections import Counter


statistics_dir = "../src/statistics/"
notebooks_dir = "../src/statistics/notebooks/"
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
    copy_statistic_files(notebooks_dir)


def get_failed_repos():

    indices = []

    for error_file in glob.glob("error/*.err"):
        #print(error_file)

        with open(error_file, "r", encoding="utf-8") as src:
            lines = src.readlines()
            lines = lines[-6:]
            if "launcher | Done in" not in lines[-1]:
                index = error_file.split("-")[-1].split(".")[0]
                indices.append(int(index))

    
    with open("final_sample_set.json", "r", encoding="utf-8") as src:
        data = json.load(src)

        failed_repos = [data[index] for index in indices]


    with open("failed_repos.json", "w", encoding="utf-8") as src:
        json.dump(failed_repos, src, indent=4, sort_keys=True)
    


if __name__ == "__main__":
    get_failed_repos()