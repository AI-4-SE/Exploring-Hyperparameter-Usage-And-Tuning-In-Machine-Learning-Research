import glob
import shutil
import json
import csv
from numpy import full
import pandas as pd
import random

from collections import Counter


statistics_dir = "../src/data/statistics/"
notebooks_dir = "../src/data/notebooks/"
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

def copy_random_statistics_files(target, n = 1000):
    files = []
    for repo in glob.glob("results/*"):
        for csv_file in glob.glob(f"{repo}/statistics/*"):
            if csv_file.endswith("params.json"):
                name = csv_file.split("/")[-1]
                print(csv_file)
                files.append(csv_file)

    random_files = random.sample(files, n)

    for f in random_files:
        name = f.split("/")[-1]
        shutil.copyfile(f, target + name)

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
    #copy_random_statistics_files(statistics_dir)

    files = glob.glob(f"{statistics_dir}/*")
    print(len(files))