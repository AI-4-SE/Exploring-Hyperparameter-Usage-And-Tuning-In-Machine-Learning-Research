import glob
import json
import random

def main():

    repo_names = set()
    repo_urls = set()
    repo_meta_data = []


    for stats_name in glob.glob("D:\\GitHub\\AI-4-SE\\ml-settings\\src\\data\\statistics\\*.json"):
        file_name = stats_name.split("\\")[-1]
        name = file_name.split("_params")[0]
        repo_names.add(name.lower())

    with open("sample_set_urls.json", "r", encoding="utf-8") as src:
        data = json.load(src)

    for url in data:
        _name = url.split("/")[-1].lower()
        if "." in _name:
            _name = _name.split(".")[0]
        for name in repo_names:
            if name == _name:
                repo_urls.add(url)
                break

    print("Repo names: ", len(repo_names))
    print("Repo Urls: ", len(repo_urls))

    urls = random.sample(repo_urls, 2000)

    with open("final_sample_set_urls.json", "w", encoding="utf-8") as dest:
        json.dump(urls, dest, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()

