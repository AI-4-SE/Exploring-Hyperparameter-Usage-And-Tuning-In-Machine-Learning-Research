#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("Please upvote :)")


# # Installations

# In[ ]:


pip install -q gitpython


# # Imports

# In[ ]:


from tqdm import tqdm
import os
import shutil
import git
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import shuffle


# # Parameters

# In[ ]:


i, n_urls = (
    0,
    10 ** 3,
)  # i is in range(10) to select a tenth of repository urls; n_urls is the total number of repository urls
i, n_urls


# # Utils

# In[ ]:


def get_filepaths():
    filepaths = []

    for dirpath, _, filenames in os.walk("repository"):
        for filename in filenames:
            if filename.endswith(".py"):  # Python files
                filepath = os.path.join(dirpath, filename)

                try:
                    strng = open(filepath).read()  # read file

                    if strng.strip() != "":  # non-empty strings
                        filepaths.append(filepath)
                except:
                    pass

    return filepaths


# In[ ]:


def move_python_files(j, filepaths):
    for filepath in filepaths:
        Path(filepath).rename(f"scripts/script_{j}.py")
        j += 1
    return j


# # Load Scripts

# In[ ]:


filepath = "../input/urls-of-top-python-github-projects/urls_of_top_python_github_projects.csv"
urls = list(np.squeeze(pd.read_csv(filepath).values))[:n_urls]
urls = shuffle(urls, random_state=0)
urls[:5], len(urls)


# In[ ]:


ranges = np.linspace(0, len(urls), 11, dtype=int)
urls = urls[ranges[i] : ranges[i + 1]]  # one tenth per notebook
urls[:5], len(urls)


# In[ ]:


urls_for_removal = [  # problematic Python repositories
    "https://github.com/owid/covid-19-data.git",
    "https://github.com/covid19india/api.git",
    "https://github.com/echen102/COVID-19-TweetIDs.git",
]

urls = [url for url in urls if url not in urls_for_removal]
urls[:5], len(urls)


# In[ ]:


j, directory = 0, "scripts"

if os.path.isdir(directory):
    shutil.rmtree(directory)

os.makedirs(directory)

if os.path.isdir("repository"):
    shutil.rmtree("repository")

for url in tqdm(urls):
    git.Repo.clone_from(url, "repository")
    filepaths = get_filepaths()
    j = move_python_files(j, filepaths)
    shutil.rmtree("repository")


# # Display

# In[ ]:


for filename in os.listdir(directory)[:2]:
    filepath = os.path.join(directory, filename)
    strng = open(filepath).read()
    print(strng)
    print(
        "*******************************************************************************************************"
    )


# # Save

# In[ ]:


get_ipython().system('tar -czf scripts.tar.gz scripts')


# In[ ]:


shutil.rmtree("scripts")

