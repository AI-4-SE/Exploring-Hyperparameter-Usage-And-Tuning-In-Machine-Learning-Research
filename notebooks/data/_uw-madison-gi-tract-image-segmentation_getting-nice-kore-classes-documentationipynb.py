#!/usr/bin/env python
# coding: utf-8

# For those people, like me, struggling to get a nice overview of the objects like Board, Shipyard, and so on, I found a nice way to create an HTML file view with links.  You can download it to your local machine and view with any browser, like so:
# #chromium-browser helpers.html
# or open it with an open-file menu item, depending on your browser.

# In[ ]:


#First find where the helpers file is
get_ipython().system('find / -name helpers.py | grep kore_fleets')


# In[ ]:


#Copy it to our working directory
get_ipython().system('cp -v /opt/conda/lib/python3.7/site-packages/kaggle_environments/envs/kore_fleets/helpers.py .')


# In[ ]:


#Now generate the HTML pydoc file
get_ipython().system('pydoc -w helpers')


# In[ ]:


from IPython.core.display import display, HTML
#HTML(filename='helpers.html')    # this displays with bad formatting, instead do
from IPython.display import IFrame
display(IFrame('helpers.html', '100%','300px'))


# You can see how to get all kinds of nice info from that, for example:
# Cell.fleet.flightplan
# or
# shipyard.max_spawn
# 

# In[ ]:


#Display the html file here or download it to your local machine and view it
get_ipython().system('cat helpers.html')


# In[ ]:




