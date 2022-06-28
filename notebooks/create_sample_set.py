import glob
import random
import pandas as pd
import shutil

NOTEBOOK_DIR = "D:\\AgileAI\\Kaggle_Notebooks\\kaggle_scripts\\*"
SAMPLE = "sample_set.csv"

def create_sample_set():
    files = glob.glob(NOTEBOOK_DIR)

    sample_set = random.sample(files, 1000)

    #print(sample_set)
    #print("len: ", len(sample_set))

    df = pd.DataFrame(sample_set, columns=["file"], index=False)

    df.to_csv("sample_set.csv")


def find_relevant():
    pass


def copy_files(target):
    df = pd.read_csv(SAMPLE)

    files  = df["file"].tolist()

    for file in files:
        name = file.split("\\")[-1]
        print(name)
        shutil.copyfile(file, target + name)

def main():
    copy_files("data/")


if __name__ == "__main__":
    main()