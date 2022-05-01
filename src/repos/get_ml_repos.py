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

target_dir = "\\\\nfs.ceph.dw.webis.de\\cephfs\\data-tmp\\2021\\liju1602\\paperswithcode_crawl"

SKLEARN_REGEX = re.compile(r"import sklearn")
SKLEARN_FROM_REGEX = re.compile(r"from sklearn[a-zA-z._]* import [a-zA-Z_]*")

TENSORFLOW_REGEX = re.compile(r"import tensorflow")
TENSORFLOW_FROM_REGEX = re.compile(r"from tensorflow[a-zA-z._]* import [a-zA-Z_]*")

PYTORCH_REGEX = re.compile(r"import torch")
PYTORCH_FROM_REGEX = re.compile(r"from torch[a-zA-z._]* import [a-zA-Z_]*")

BASE = "/home/ssimon/GitHub/ml-settings/"
SAMPLE_SET = "data/sample_set.txt"

def create_sample_set(number):
    for _, _, files in os.walk(target_dir):
        sample = set(random.sample(files, number))
        
    with open("./data/sklearn/sample_set.txt", "w") as sample_set:
        for file in sample:
            sample_set.write(file + "\n")

def read_samples(file_path):
    with open(BASE + file_path, "r") as sample_set:
        lines = sample_set.readlines()
        lines = [line.replace("\n", "") for line in lines]
    return lines

def find_ml_repos(import_regex, import_from_regex, file_path):
    samples = read_samples(BASE + SAMPLE_SET)

    with open(BASE + file_path, "w") as source:
        for subdir, _ , files in os.walk(target_dir):
            for file in files:
                is_ml_repo = False
                if file in samples:
                    print("Processing: " + file)
                    tar = tarfile.open(os.path.join(subdir, file))
                    for member in tar.getmembers():
                        if member.name.endswith(".py"):
                            filepath = subdir + os.sep + member.name
                            filepath = filepath.replace("/", "\\")
                            try:
                                f=tar.extractfile(member)
                                for line in f.readlines():
                                    if import_regex.search(str(line)):
                                        source.write(file + "\n")
                                        is_ml_repo = True
                                        break
                                    if import_from_regex.search(str(line)):
                                        source.write(file + "\n")
                                        is_ml_repo = True
                                        break
                            except (AttributeError, KeyError):
                                print("skipped: ", filepath)
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
    repos = []
    with open(BASE + url, "r", encoding="utf-8") as source:
        for line in source.readlines():
            data = line.split(",")
            repos.append((data[0], data[1].replace("\n", "")))

    print(len(repos))
    print(repos)

    with open(BASE + url_final, "w") as source:
        for subdir, _ , files in os.walk(target_dir):
            for repo in repos:
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
                                source.write(f"{repo[0]},{repo[1]}" + "\n")
                            except Exception as error:
                                # Drop repository if not written in Python3
                                print("Drop repo: ", repo[0])


if __name__ == "__main__":
    # create_sample_set(1000), already done

    sklearn_path = "/data/sklearn/sklearn_sample.txt"
    sklearn_url = "/data/sklearn/sklearn_sample_url.csv"
    sklearn_url_final = "/data/sklearn_sample_url_final.csv"
    tensorflow_path = "/data/tensorflow/tensorflow_samples.txt"
    tensorflow_url = "/data/tensorflow/tensorflow_samples_url.csv"
    tensorflow_url_final = "/data/tensorflow/tensorflow_samples_url_final.csv"
    pytorch_path = "/data/pytorch/pytorch_samples.txt"
    pytorch_url = "/data/pytorch/pytorch_samples_url.csv"
    pytorch_url_final = "/data/pytorch/pytorch_samples_url_final.csv"

    print(os.path.dirname(os.path.abspath(__file__)))

    #find_ml_repos(SKLEARN_REGEX, SKLEARN_FROM_REGEX, sklearn_path)
    #find_ml_repos(TENSORFLOW_REGEX, TENSORFLOW_FROM_REGEX, tensorflow_path)
    #find_ml_repos(PYTORCH_REGEX, PYTORCH_FROM_REGEX, pytorch_path)
    #get_urls(pytorch_path, pytorch_url)
    check_python_version(sklearn_url, sklearn_url_final)
