from typing import List
from aquarel import load_theme
import matplotlib.pyplot as plt
import json

hyperparameter_importance = "../data/dblp/hyperparameter_importance.json"
hyperparameter_tuning = "../data/dblp/hyperparameter_tuning.json"
hyperparameter_optimization = "../data/dblp/hyperparameter_optimization.json"

files = [hyperparameter_importance, hyperparameter_tuning, hyperparameter_optimization]

def get_hits_by_year(year: str) -> int:
    matches = set()

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as src:
            data = json.load(src)
            results = data["result"]
            hits = results["hits"]

            for hit in hits["hit"]:
                info = hit["info"]

                if year == info["year"]:
                    matches.add(info["title"])

                

    return len(matches)

def create_histogram():
    
    data = [
        get_hits_by_year("2015"), 
        get_hits_by_year("2016"), 
        get_hits_by_year("2017"),
        get_hits_by_year("2018"),
        get_hits_by_year("2019"),
        get_hits_by_year("2020"),
        get_hits_by_year("2021"),
        get_hits_by_year("2022"),
    ]

    labels = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]
    
    #with load_theme("boxy_light"):
    fig, ax = plt.subplots()

    ax.bar(labels, data)

    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Papers')
    ax.grid(axis='x')

    plt.grid()

    #plt.tight_layout(pad=0)
    #plt.show()
    plt.savefig("papers_per_year.svg",bbox_inches='tight',  pad_inches=0)

def main():
    print("Matches 2015: ", get_hits_by_year("2015"))
    print("Matches 2016: ", get_hits_by_year("2016"))
    print("Matches 2017: ", get_hits_by_year("2017"))
    print("Matches 2018: ", get_hits_by_year("2018"))
    print("Matches 2019: ", get_hits_by_year("2019"))
    print("Matches 2020: ", get_hits_by_year("2020"))
    print("Matches 2021: ", get_hits_by_year("2021"))
    print("Matches 2022: ", get_hits_by_year("2022"))


if __name__ == "__main__":
    create_histogram()
