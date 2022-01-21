import os
import re
import random
import tarfile
import csv
import json
import pandas as pd

target_dir = "\\\\nfs.ceph.dw.webis.de\\cephfs\\data-tmp\\2021\\liju1602\\paperswithcode_crawl"


def create_sample_set(number):
    for _, _, files in os.walk(target_dir):
        sample = set(random.sample(files, number))
        
    with open("./data/sklearn/sample_set.txt", "w") as sample_set:
        for file in sample:
            sample_set.write(file + "\n")

def read_samples(file_path):
    with open(file_path, "r") as sample_set:
        lines = sample_set.readlines()
        lines = [line.replace("\n", "") for line in lines]
    return lines

def find_scikit_learn_repos():
    import_regex = re.compile(r"import sklearn")
    import_from_regex = re.compile(
            r"from sklearn[a-zA-z._]* import [a-zA-Z_]*"
        )
    samples = read_samples("./data/sklearn/sample_set.txt")

    with open("./data/sklearn/sklearn_sample.txt", "w") as source:
        for subdir, _ , files in os.walk(target_dir):
            for file in files:
                is_scikit_learn_repo = False
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
                                        is_scikit_learn_repo = True
                                        break
                                    if import_from_regex.search(str(line)):
                                        source.write(file + "\n")
                                        is_scikit_learn_repo = True
                                        break
                            except (AttributeError, KeyError):
                                print("skipped: ", filepath)
                        if is_scikit_learn_repo:
                            break

def get_urls():
    tar_files = read_samples("./data/sklearn/sklearn_sample.txt")
    print("tar files len: ", len(tar_files))

    with open("./data/sklearn/sklearn_sample_final.csv", "w", newline = '\n') as file:
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


if __name__ == "__main__":
    # create_sample_set(1000), already done
    # find_scikit_learn_repos(), already done
    # get_urls(), already done

    print(dir)

    if os.path.exists(dir):
        print("true")
    else:
        print("false")
