#!/usr/bin/env python
# coding: utf-8

# # Going from superpermutation ➡️ permutation format
# When working on optimisation problems, you may want to use an existing solution (e.g. a `submission.csv`) as an initial starting point for another round of optimisation.
# 
# The class below does the following:
# * Split the string back into the 7 emoji "cities"
# * If a city has a wildcard, try and figure out what the city should be
# 
# Note that the code is a bit rough and can certainly be optimised. There may also be some edge cases that will cause the wildcard lookup to fail. Any feedback is appreciated!

# In[ ]:


import pandas as pd
import itertools


# In[ ]:


class ExtractCities:
    def __init__(
        self, letters=["🎅", "🤶", "🦌", "🧝", "🎄", "🎁", "🎀"], wildcard="🌟", verbose=True
    ):
        self.letters = letters
        self.wildcard = wildcard
        self.verbose = verbose
        self.N = len(letters)
        self.perms = itertools.permutations(letters, self.N)
        self.perms = ["".join(list(p)) for p in self.perms]

    def __call__(self, df_path):
        df = pd.read_csv(df_path)
        strings = [x for x in df["schedule"].tolist()]
        strings_perm_fmt = [self.extract_permutations(s) for s in strings]
        all_cities = [item for sublist in strings_perm_fmt for item in sublist]

        taken = [s for s in all_cities if s in self.perms]  # Used permutations
        candidates = [p for p in self.perms if p not in taken]  # Unused permutations
        wcs = [s for s in all_cities if self.wildcard in s]  # Cities with wildcards

        mappings = {}
        for i, s in enumerate(wcs):
            matched = False
            for c in candidates:
                n_wildcards = s.count(self.wildcard)
                if self.similarity(s, c) == self.N - n_wildcards:
                    mappings[s] = c
                    matched = True
                    if self.verbose:
                        print(i, "Wildcard string", s, "matches", c, "from cands")

            for p in self.perms:
                if self.similarity(s, p) == self.N - n_wildcards and not matched:
                    mappings[s] = p
                    if self.verbose:
                        print(i, "Wildcard string", s, "matches", p, "from perms")
                        
        strings_final = []
        for string in strings_perm_fmt:
            strings_final.append([mappings[x] if self.wildcard in x else x for x in string])
            
        return strings_final

    def similarity(self, string1, string2):
        score = 0
        for a, b in zip(string1, string2):
            if a == b:
                score += 1
        return score

    def extract_permutations(self, string):
        permutations = []
        for k in range(len(string) - self.N + 1):
            s = string[k : k + self.N]

            n_wildcards = s.count(self.wildcard)

            if n_wildcards == 0:
                is_perm = len(set(s)) == self.N
            else:
                is_perm = len(set(s)) == self.N - n_wildcards + 1

            if is_perm and s not in permutations:
                permutations.append(s)

        return permutations


# In[ ]:


ec = ExtractCities()
out = ec("/kaggle/input/santa-2021-baseline-and-optimization-ideas/submission.csv")


# In[ ]:


len(out)


# In[ ]:


# Each string has now been broken into 7-character strings
out[0][:10]


# In[ ]:


# Check if any wildcards still remain
for string in out:
    for s in string:
        if "🌟" in s:
            print(s)


# In[ ]:




