import os
import re
import random
import tarfile
import csv
import json
import pandas as pd
import git
import ast
from git import Repo
import bz2
import requests


target_dir = "\\\\nfs.ceph.dw.webis.de\\cephfs\\data-tmp\\2021\\liju1602\\paperswithcode_crawl"

SKLEARN_REGEX = re.compile(r"import sklearn")
SKLEARN_FROM_REGEX = re.compile(r"from sklearn[a-zA-z._]* import [a-zA-Z_]*")

TENSORFLOW_REGEX = re.compile(r"import tensorflow")
TENSORFLOW_FROM_REGEX = re.compile(r"from tensorflow[a-zA-z._]* import [a-zA-Z_]*")

PYTORCH_REGEX = re.compile(r"import torch")
PYTORCH_FROM_REGEX = re.compile(r"from torch[a-zA-z._]* import [a-zA-Z_]*")

directory_parts = os.path.dirname(os.path.abspath(__file__)).split("\\")
directory_parts = directory_parts[:-2]
BASE = "\\".join(directory_parts)
SAMPLE_SET = "\\src\\repos\\sample_set_5000.txt"

def create_sample_set(number):
    for _, _, files in os.walk(target_dir):
        sample = set(random.sample(files, number))
        
    with open("sample_set_5000.txt", "w") as sample_set:
        for file in sample:
            sample_set.write(file + "\n")

def read_samples(file_path):
    with open(BASE + file_path, "r") as sample_set:
        lines = sample_set.readlines()
        lines = [line.replace("\n", "") for line in lines]
    return lines

def check_imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for package in node.names:
                module = package.name.split(".")[0]
                if module in ("sklearn", "tensorflow", "torch"):
                    return True
                else:
                    return False

        if isinstance(node, ast.ImportFrom):
            for package in node.names:
                if node.module:
                    module = node.module.split(".")[0]
                    if module in ("sklearn", "tensorflow", "torch"):
                        return True
                    else:
                        return False


def find_ml_repos(file_path):
    samples = read_samples(SAMPLE_SET)
    print(BASE + file_path)
    with open(BASE + file_path, "w") as source:
        for subdir, _ , files in os.walk(target_dir):
            for file in files:
                is_ml_repo = False
                if file in samples:
                    tar = tarfile.open(os.path.join(subdir, file))
                    for member in tar.getmembers():
                        if member.name.endswith(".py"):
                            filepath = subdir + os.sep + member.name
                            filepath = filepath.replace("/", "\\")
                            try:
                                f=tar.extractfile(member)
                                code_str = f.read()
                                tree = ast.parse(code_str)
                                if check_imports(tree):
                                    source.write(file + "\n")
                                    is_ml_repo = True
                                    break
                            except (AttributeError, KeyError, SyntaxError, IndentationError, Exception):
                                continue
                        if is_ml_repo:
                            break

def get_urls(sample_path, url_path):
    tar_files = read_samples(sample_path)
    print("tar files len: ", len(tar_files))

    with open(BASE + url_path, "w", newline = '\n') as file:
        writer = csv.DictWriter(file, fieldnames=["tar_filename", "repo_url"])
        writer.writeheader()
        with open('paperswithcode_repos_220102.jsonl', 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)
            if result["tar_filename"] in tar_files:
                data = {
                    "tar_filename": result["tar_filename"],
                    "repo_url": result["repo_url"],
                }
                writer.writerow(data)


def check_python_version(url, url_final):
    samples = read_samples(SAMPLE_SET)
    drop_counter = 0
    repos = []
    with open(BASE + url, "r", encoding="utf-8") as source:
        for line in source.readlines():
            data = line.split(",")
            repos.append((data[0], data[1].replace("\n", "")))

    with open(BASE + url_final, "w") as source:
        source.write("tar_filename,repo_url" + "\n")
        for subdir, _ , _ in os.walk(target_dir):
            for repo in repos:
                dropped_repo = False
                if repo[0] in samples:
                    tar = tarfile.open(os.path.join(subdir, repo[0]))
                    for member in tar.getmembers():
                        if member.name.endswith(".py"):
                            try:
                                # Check Python Version
                                filepath = subdir + os.sep + member.name
                                filepath = filepath.replace("/", "\\")
                                f=tar.extractfile(member)
                                code_str = f.read()
                                tree = ast.parse(code_str)
                                
                            except Exception as error:
                                # Drop repository if not written in Python3
                                print("Dropped repo: ", repo[0], error)
                                dropped_repo = True
                        if dropped_repo:
                            drop_counter +=1
                            break
                    if not dropped_repo:
                        source.write(f"{repo[0]},{repo[1]}" + "\n")
    
    print("Drop Counter: ", drop_counter)


def get_sample_set():
    sample_data_set = []
    with bz2.BZ2File("../../data/5000/pswc_repos_papers.jsonl.bz2") as file:
        json_list = list(file)

    sample_data = random.sample(json_list, 5000)

    for json_str in sample_data:
        data = json.loads(json_str)
        if any(x in data["code_stats"]["imports"] for x in ("tensorflow", "sklearn", "torch")) and data["papers"][0]["paper_title"] and data["repo_url"]:
            sample_data_set.append(data)

    print(len(sample_data_set))
    with open("sample_set.json", "w", encoding="utf-8") as dest:
        json.dump(sample_data_set, dest, indent=4, sort_keys=True)


def check_url():
    final_data_set = []

    with open("sample_set.json", "r", encoding="utf-8") as src:
        data = json.load(src)

    for item in data:
        url = item["repo_url"]
        request = requests.get(url)
        try:
            if request.status_code == 200:
                final_data_set.append(item)
            else:
                print('Url does not exist: ', url)
        except Exception:
            print("Exception occurred for: ", url)

    print(len(final_data_set))
    with open("final_sample_set.json", "w", encoding="utf-8") as dest:
        json.dump(final_data_set, dest, indent=4, sort_keys=True)

def get_urls():
    urls = []

    with open("sample_set.json", "r", encoding="utf-8") as src:
        data = json.load(src)

    data = random.sample(data, 2500)

    for item in data:
        urls.append(item["repo_url"])

    with open("sample_set_urls.json", "w", encoding="utf-8") as dest:
        json.dump(urls, dest, indent=4, sort_keys=True)  



def get_metadata():
    metadata = []
    
    with open("final_sample_set_urls.json", "r", encoding="utf-8") as src:
        data = json.load(src)


    repo_papers = []
    with bz2.BZ2File("pswc_repos_papers.jsonl.bz2") as bz2_file:
        for line in bz2_file:
            repo_papers.append(json.loads(line))

    for url in data:
        try:
            repo_data = next(filter(lambda x : x["repo_url"] == url, repo_papers))
            metadata.append(repo_data)
        except StopIteration:
            print("No data found for: ", url)


    with open("final_metadata_set.json", "w", encoding="utf-8") as dest:
        json.dump(metadata, dest, indent=4, sort_keys=True)


if __name__ == "__main__":
    #create_sample_set(5000)
    #find_ml_repos("\\src\\repos\\ml_samples_5000.txt")
    #get_urls("\\src\\repos\\ml_samples_5000.txt", "\\src\\repos\\ml_samples_url_5000.csv")

    #sklearn_path = "\\data\\sklearn\\sklearn_sample.txt"
    #sklearn_url = "\\data\\sklearn\\sklearn_sample_url.csv"
    #sklearn_url_final = "\\data\\sklearn\\sklearn_sample_url_final.csv"
    #tensorflow_path = "\\data\\tensorflow\\tensorflow_samples.txt"
    #tensorflow_url = "\\data\\tensorflow\\tensorflow_samples_url.csv"
    #tensorflow_url_final = "\\data\\tensorflow\\tensorflow_samples_url_final.csv"
    #pytorch_path = "\\data\\pytorch\\pytorch_samples.txt"
    #pytorch_url = "\\data\\pytorch\\pytorch_samples_url.csv"
    #pytorch_url_final = "\\data\\pytorch\\pytorch_samples_url_final.csv"

    #find_ml_repos(SKLEARN_REGEX, SKLEARN_FROM_REGEX, sklearn_path)
    #find_ml_repos(TENSORFLOW_REGEX, TENSORFLOW_FROM_REGEX, tensorflow_path)
    #find_ml_repos(PYTORCH_REGEX, PYTORCH_FROM_REGEX, pytorch_path)
    #get_urls(pytorch_path, pytorch_url)
    #check_python_version(pytorch_url, pytorch_url_final)

    with open("final_metadata_set.json", "r", encoding="utf-8") as src:
        data = json.load(src)
        print(len(data))


    #get_metadata()
