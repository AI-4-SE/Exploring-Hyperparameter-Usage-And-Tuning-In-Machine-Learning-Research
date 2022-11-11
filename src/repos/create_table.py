import json
import pandas as pd

def get_paper_data():
    projects = []
    titles = []
    authors = []
    domain = []

    with open("final_metadata_set.json", "r", encoding="utf-8") as src:
        data = json.load(src)

    for item in data:
        project_name = item["repo_url"].split("/")[-1]
        projects.append(project_name)
        paper_data = item["paper"]
        if paper_data["title"]:
            titles.append(paper_data["title"])
        else:
            paper_data = item["papers"][0]
            titles.append(paper_data["paper_title"])

    df = pd.DataFrame()
    df["project name"] = projects
    df["titles"] = titles
    df.to_csv("papers.csv")


if __name__ == "__main__":
    get_paper_data()