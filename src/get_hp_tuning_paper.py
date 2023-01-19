import pandas as pd
import glob
import json
from typing import List

file_path = "../../data/paper_analysis/metadata-with-categories-cleaned.csv"

def get_hp_papers() -> pd.DataFrame:

    df = pd.read_csv(file_path)

    df_hp = df[df["hyperparameter"] == "yes"]

    
    return df_hp

def get_statistics_files(df: pd.DataFrame) -> List[str]:

    final_statistics_files = []

    statistic_files = glob.glob("../../data/statistics/**")
    #print(len(statistic_files))

    urls = df["repo_url"].tolist()

    for x in statistic_files:
        tmp = x.split("\\")[-1].split("_params.json")[0]
        for url in urls:
            url = url.split("/")[-1]
            if url == tmp:
                final_statistics_files.append(x.split("\\")[-1])

    #print(final_statistics_files)
    #print(len(final_statistics_files))

    with open("../../data/repos_hyperparameter_tuning.json", "w", encoding="utf-8") as dest:
        json.dump(final_statistics_files, dest, sort_keys=True, indent=4)


if __name__ == "__main__":
    df = get_hp_papers()
    get_statistics_files(df)

