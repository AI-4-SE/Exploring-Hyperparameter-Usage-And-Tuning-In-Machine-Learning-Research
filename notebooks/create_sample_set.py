import glob
import random
import pandas as pd
import shutil
import ast

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


def copy_files(source, target):
    df = pd.read_csv(source)

    files  = df["file"].tolist()

    for file in files:
        name = file.split("\\")[-1]
        print(name)
        shutil.copyfile(file, target + name)



def check_imports():
    notebooks = set()

    df = pd.read_csv("sample_set.csv")
    files = df["file"]

    for file_name in files:
        with open(file_name, "r", encoding="utf-8") as src:
            try:
                tree = ast.parse(src.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for package in node.names:
                            module = package.name.split(".")[0]
                            if module in ("sklearn", "torch", "tensorflow"):
                                notebooks.add(file_name)
                                continue

                    if isinstance(node, ast.ImportFrom):
                        for package in node.names:
                            if node.module:
                                module = node.module.split(".")[0]
                                if module in ("sklearn", "torch", "tensorflow"):
                                    notebooks.add(file_name)
                                    continue

            except SyntaxError:
                print("%s could not be parsed.", file_name)


    final_df = pd.DataFrame(data=notebooks)
    final_df.to_csv("sample_set_final.csv")






def main():
    #check_imports()
    copy_files("sample_set_final.csv", "D:\\GitHub\\AI-4-SE\\final_notebooks")

if __name__ == "__main__":
    main()