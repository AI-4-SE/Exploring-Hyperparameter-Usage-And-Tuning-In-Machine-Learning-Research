from calendar import c
import json
import random
import pandas as pd

def main():

    with open("final_metadata_set.json", "r", encoding="utf-8") as src:
        data = json.load(src)
        #print(data)
        print(len(data))

        data = data[1500:]
        print(len(data))  

        samples = random.sample(data, 100)
        print(len(samples))


        for item in samples:
            paper = item["paper"]
            if paper["title"]:
                print(paper["title"])
            else:
                paper = item["papers"][0]
                print(paper["paper_title"])


def test():
    df = pd.read_csv("metadata.csv")

    print(df.head())
    
    df_first = df[:1500]
    df_first = df_first.sample(100)

    df_last = df[1500:]
    df_last = df_last.sample(100)

    df_cross = pd.concat([df_first, df_last])
    df_cross.to_csv("cross-valdiation.csv")


if __name__ == "__main__":
    #main()
    test()