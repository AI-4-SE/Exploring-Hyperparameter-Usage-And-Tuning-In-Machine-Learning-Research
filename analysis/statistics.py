import glob
import shutil
import json
import csv
from numpy import full
import pandas as pd
import random
import numpy as np

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

def copy_random_statistics_files(target, n = 2000):
    files = []
    for repo in glob.glob("results/*"):
        for csv_file in glob.glob(f"{repo}/statistics/*"):
            if csv_file.endswith("params.json"):
                name = csv_file.split("/")[-1]
                #print(csv_file)
                files.append(csv_file)

    print(len(files))

    #random_files = random.sample(files, n)

    #for f in random_files:
    #    name = f.split("/")[-1]
    #    shutil.copyfile(f, target + name)


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
    

def test():
    
    with open("final_sample_set_urls.json", "r", encoding="utf-8") as src:
        urls = json.load(src)

    

    url_repo_names = [x.split("/")[-1] for x in urls]

    repos = []
    for repo_name in glob.glob("results/**"):
        name = repo_name.split("/")[-1]
        repos.append(name)

    
    print("Len Urls: ", len(urls))
    print("Len Url repo names: ", len(url_repo_names))
    print("Len Repos: ", len(repos)) 

    url_repo_names = [x.lower() for x in url_repo_names]
    repos = [x.lower() for x in repos]

    test = []

    for name in url_repo_names:
        if name in repos:
            test.append(name)

    
    print(len(test))


def get_external_config_files():

    names = []
    count_config_files = []
    config_files = []


    for repo in glob.glob("results/**"):
        tmp = set()
        found_line = False
        for log_file in glob.glob(f"{repo}/*.log"):
            name = log_file.split("/")[-1].split(".")[0]
            names.append(name)

            with open(log_file, "r", encoding="utf-8") as src:
                lines = src.readlines()
                for line in lines[-3:]:
                    if "No Config files found." in line:
                        count_config_files.append(0)
                        config_files.append([])
                        found_line = True
                    elif "Config files:" in line:
                        files = line.split("Config files:")[-1]
                        files = files.split(", ")
                        for file in files:
                            if any(x in files for x in [".github", "/paper/", ".pre-commit"]):
                                continue
                            if file.endswith((".yaml", ".yml", ".json")):
                                if "/config" in file:
                                    tmp.add(file.strip())
                    
                        count_config_files.append(len(tmp))
                        config_files.append(tmp)
                        found_line = True                        
                if not found_line:
                    count_config_files.append(0)
                    config_files.append(tmp)
                    print(name)

    df = pd.DataFrame()
    df["Name"] = names
    df["Count"] = count_config_files
    df["Files"] = config_files

    df.to_csv("config_files_count.csv")


def get_stats():
    df = pd.read_csv("config_files_count.csv")

    print(df['Count'].value_counts())
    print(len(df[df["Count"] == 0]))
    print(len(df[df["Count"] > 0]))



if __name__ == "__main__":
    #copy_random_statistics_files(statistics_dir)
    #copy_statistic_files(statistics_dir)

    #files = glob.glob(f"{statistics_dir}/*")
    #print(len(files))

    get_external_config_files()
    get_stats()