import json
from collections import Counter
import pandas as pd

def get_category(task, category_data):
    x = []

    for item in category_data:
        for _task in item["tasks"]:
            if task == _task:
                x.append(item["category"])
    return x

def assign_categories():
    with open("categories_all.json", "r", encoding="utf-8") as src:
        category_data = json.load(src)

    with open("final_metadata.json", "r", encoding="utf-8") as src:
        paper_data = json.load(src)

    papers_with_categories = []
    greater_counter = 0
    equal_counter = 0
    less_counter = 0

    for item in paper_data:
        all_categories = []
        paper = item["paper"]
        if paper:
            tasks = paper["tasks"]
            if tasks:
                for task in tasks:
                    categories = get_category(task, category_data)

                    for x in categories:
                        all_categories.append(x)

                category_counter = Counter(all_categories)

                
                x = [item[0] for item in category_counter.most_common() if item[1] == category_counter.most_common(1)[0][1]]


                if len(x) > 1:
                    greater_counter += 1

                if len(x) == 1:
                    equal_counter += 1

                if len(x) == 0:
                    less_counter += 1
           
                if x:
                    paper_category = {
                        "tar_filename": item["tar_filename"],
                        "abstract": paper["abstract"],
                        "body": paper["body"],
                        "file_name": paper["filename"],
                        "title": paper["title"],
                        "category": x
                    }
                    papers_with_categories.append(paper_category)
        
    print("Greater Counter: ", greater_counter)
    print("Equal Counter: ", equal_counter)
    print("Less Counter: ", less_counter)
    
    with open("papers_with_categories.json", "w", encoding="utf-8") as dest:
        json.dump(papers_with_categories, dest, sort_keys=True, indent=4)

def add_categories_to_csv():
    with open("papers_with_categories.json", "r", encoding="utf-8") as src:
        data = json.load(src)

    df = pd.read_csv("metadata-cleaned.csv")

    tar_filenames = df["tar_filename"].tolist()

    categories = []

    for tar_file in tar_filenames:
        match = list(filter(lambda x: x["tar_filename"] == tar_file, data))
        #print("len match: ", len(match))
        if len(match) > 1:
            print(tar_file)

        if match:

            categories.append(", ".join(match[0]["category"]))
        else:
            categories.append("Unknown")

    df["categories"] = categories
    df.to_csv("metadata-with-categories.csv")

def main():
    df = pd.read_csv("metadata-with-categories-cleaned.csv")

    categories = []

    for _, row in df.iterrows():
        parts = str(row["categories"]).split(",")
        for part in parts:
            categories.append(part.strip())


    counter = Counter(categories)
    for x in counter.most_common():
        print(x)


if __name__ == "__main__":
    #assign_categories()
    #add_categories_to_csv()
    main()