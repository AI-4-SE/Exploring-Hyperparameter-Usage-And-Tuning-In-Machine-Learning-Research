#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -U sec-edgar-downloader')


# In[ ]:


from sec_edgar_downloader import Downloader

# Initialize a downloader instance. If no argument is passed
# to the constructor, the package will download filings to
# the current working directory.
dl = Downloader("/kaggle/working/")


# In[ ]:


dl.get("20-F", "DEO", amount=1)


# In[ ]:


from bs4 import BeautifulSoup
with open('/kaggle/working/sec-edgar-filings/DEO/20-F/0000835403-21-000011/filing-details.html') as f:
    html = f.read()

soup = BeautifulSoup(html,'html5lib')
cans = soup.findAll('div')
for can in cans:
    print(">",can.text)

