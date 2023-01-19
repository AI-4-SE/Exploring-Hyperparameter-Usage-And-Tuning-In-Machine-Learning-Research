import subprocess
import os
import glob
import subprocess
import sys
from git import Repo
import json

# The folder where we store our results.
EVALUATION_FOLDER = "out"

NOTEBOOK_REPO = ["https://github.com/simisimon/kaggle-notebooks"]

def get_repos():
    file_name = "final_sample_set_urls.json"

    with open(file_name, "r", encoding="utf-8") as src:
        data = json.load(src)
    
    return data


def get_repo_name_from_url(url):
    """
    Analyze a repository with CfgNet.
    :param url: URL to the repository
    :return: Repository name
    """
    repo_name = url.split("/")[-1]
    #repo_name = repo_name.split(".")[0]
    return repo_name


def process_repo(url):
    """
    Analyze a repository with CfgNet.
    :param url: URL to the repository
    :param commit: Hash of the lastest commit that should be analyzed
    :param ignorelist: List of file paths to ignore in the analysis
    """
    repo_name = get_repo_name_from_url(url)
    repo_folder = EVALUATION_FOLDER + "/" + repo_name
    results_folder = EVALUATION_FOLDER + "/results/" + repo_name
    abs_repo_path = os.path.abspath(repo_folder)

    # Cloning repository
    Repo.clone_from(url, repo_folder)

    # Init repository
    subprocess.run(
        f"cfgnet init -m {abs_repo_path}", shell=True, executable="/bin/bash"
    )

    # Copy results into result folder
    subprocess.run(["cp", "-r", repo_folder + "/.cfgnet", results_folder])

    # Remove repo folder
    remove_repo_folder(repo_folder)


def remove_repo_folder(repo_name):
    """Remove the cloned repository."""
    if os.path.exists(repo_name):
        subprocess.run(["rm", "-rf", repo_name])


def main():
    """Run the analysis."""
    # create evaluation folder
    if os.path.exists(EVALUATION_FOLDER):
        subprocess.run(["rm", "-rf", EVALUATION_FOLDER])
    subprocess.run(["mkdir", "-p", EVALUATION_FOLDER + "/results"])

    repos = get_repos()

    index = int(sys.argv[1])
    process_repo(repos[index])

if __name__ == "__main__":
    main()
