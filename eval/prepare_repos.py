import bz2
import json
import random

sklearn_file = "../data/sklearn/sklearn_sample_url_final.csv"
tf_file = "../data/tensorflow/tensorflow_samples_url_final.csv"
torch_file = "../data/pytorch/pytorch_samples_url_final.csv"

new_files = "../data/5000/ml_samples_url_5000.csv"

files = [new_files]


def get_urls():
    urls = set()

    for file_name in files:
        with open(file_name, "r", encoding="utf-8") as src:
            for line in src.readlines():
                if "tar_filename" in line or "repo_url" in line:
                    continue
                else:
                    url = line.split(",")[-1].strip()
                    urls.add(url)

    return urls


def get_repo_metadata():
    data = []
    final_data = []

    with bz2.BZ2File("../data/5000/pswc_repos_papers.jsonl.bz2") as file:
        for line in file:
            data.append(json.loads(line))


    urls = get_urls()

    for item in data:
        if any(item["repo_url"] == url for url in urls):
            final_data.append(item)


    with open("../data/5000/final_metadata_5000.json", "w", encoding="utf-8") as src:
        json.dump(final_data, src, indent=4, sort_keys=True)


def get_final_sample_set():

    urls = []

    with open("../data/5000/final_metadata_5000.json", "r", encoding="utf-8") as src:
        data = json.load(src)

    random_data = random.sample(data, 1000)

    with open("../data/5000/final_metadata_1000.json", "w", encoding="utf-8") as src:
        json.dump(random_data, src, indent=4, sort_keys=True)
    
    for item in random_data:
        urls.append(item["repo_url"])

    with open("final_sample_set.json", "w", encoding="utf-8") as src:
        json.dump(urls, src, indent=4, sort_keys=True)
    


if __name__ == "__main__":
    get_final_sample_set()