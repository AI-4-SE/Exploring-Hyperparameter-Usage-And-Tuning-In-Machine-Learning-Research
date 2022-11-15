import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from domains import computer_science

# identify domains of the paper
# calculate number of papers that tune hyperparameters
# calculate number of papers that train model parameters
# calculate number of papers that report final parameter values according to the previous calculated stats
# get techniques


def main():
    df = pd.read_csv("metadata-cleaned.csv")

    hp_yes = np.sum(df["hyperparameter"] == "no")
    hp_no = np.sum(df["hyperparameter"] == "yes")

    print(hp_yes, hp_no)
    print(hp_yes + hp_no)

    mp_yes = np.sum(df["model parameters"] == "no")
    mp_no = np.sum(df["model parameters"] == "yes")

    print(mp_yes, mp_no)
    print(mp_yes + mp_no)

    #print(df["hyperparameter"].unique())
    #print(df["model parameters"].unique())
    #print(df["final values"].unique())
    #print(df[df['model parameters'].isna()])
    #print(df[df['model parameters'] == "Adam"])
    #print(df[df['final values'] == "no all"]) 

    data = []

    

    for index, row in df.iterrows():
        categories= str(row["arxiv_categories"])
        categories = categories.split(" ")
        for x in categories:
            data.append(x)
        
    #counter_data = Counter(data)
    #print(counter_data)


    data_replaced = [computer_science[x] if x in computer_science else "None" for x in data]
    counter_data_cleaned = Counter(data_replaced)
    print(counter_data_cleaned)

    #plt.bar(counter_data_cleaned.keys(), counter_data_cleaned.values())
    #plt.show()
    
    

    counter = 0
    for _, row in df.iterrows():
        if row["model parameters"] == "no" and row["hyperparameter"] == "no":
            counter += 1
            

    #print("Counter: ", counter)
    #df_no = df[df["model parameters"] == "no"]
    #df_no = df_no[df_no["hyperparameter"] == "no"]
    #df_no.to_csv("test.csv")


if __name__ == "__main__":
    main()