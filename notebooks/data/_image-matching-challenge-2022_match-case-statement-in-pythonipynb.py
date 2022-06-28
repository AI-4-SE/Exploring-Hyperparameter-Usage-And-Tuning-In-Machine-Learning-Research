#!/usr/bin/env python
# coding: utf-8

# # Match-Case Statement In Python

# ## **Match-Case Statement was Introduced in version Python 3.10**

# In[ ]:


# check is match-case statement supported

import sys

if sys.version_info >= (3, 10):
    print("Greater than 3.10")
else:
    print(f"current version is: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}. and match-case is not supported yet!")

