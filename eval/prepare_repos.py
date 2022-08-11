import glob
import json

def main():

    repo_names = set()
    repo_urls = set()
    repo_meta_data = []


    for dir_name in glob.glob("results/*"):
        for stat_file in glob.glob(f"{dir_name}/statistics/*.json"):
            name = stat_file.split("/")[-1]
            name = name.split("_params")[0]
            repo_names.add(name)

    with open("sample_set_urls.json", "r", encoding="utf-8") as src:
        data = json.load(src)

    for url in data:
        _name = url.split("/")[-1]
        for name in repo_names:
            if name == _name:
                repo_urls.add(url)
                break
            


    print(len(repo_names))
    print(len(repo_urls))

    # TODO: Get metadata for final repositories
    

if __name__ == "__main__":
    main()

