#!/usr/bin/env python
# coding: utf-8

# # Rock Paper Scissors - De Bruijn Sequence
# 
# This implements a [De Bruijn Sequence](https://en.wikipedia.org/wiki/De_Bruijn_sequence) of moves.
# 
# This is a deterministic move order bot, which will play every move an equal number of times, which should be equlivant a Nash Equlibrium random bot
# 
# The idea however is to trap agents looking for statistical patterns that don't actually exist

# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', "import re\nimport random\nimport pydash\nfrom itertools import combinations_with_replacement\n\nactions = list(combinations_with_replacement([2,1,0,2,1,0],3)) * 18\n# random.shuffle(actions)\nprint('len(actions)',len(actions))\nprint(actions)\nactions = pydash.flatten(actions)\n\n# observation   =  {'step': 1, 'lastOpponentAction': 1}\n# configuration =  {'episodeSteps': 10, 'agentTimeout': 60, 'actTimeout': 1, 'runTimeout': 1200, 'isProduction': False, 'signs': 3}\ndef kaggle_agent(observation, configuration):    \n    action = actions[observation.step] % configuration.signs\n    return int(action)\n")


# In[ ]:


get_ipython().run_line_magic('run', "-i 'submission.py'")


# In[ ]:


from kaggle_environments import make
import random

env = make("rps", configuration={"episodeSteps": 20}, debug=True)
env.run(["submission.py", lambda obs, conf: random.randint(0, 2)])
# env.run(["submission.py", rock_agent])
print(env.render(mode="ipython", width=600, height=600))


# # Further Reading
# 
# This notebook is part of a series exploring Rock Paper Scissors:
# - [Rock Paper Scissors - PI Bot](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-pi-bot)
# - [Rock Paper Scissors - De Bruijn Sequence](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-de-bruijn-sequence)
# - [Rock Paper Scissors - Random Agent](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-random-agent)
# - [Rock Paper Scissors - Weighted Random Agent](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-weighted-random-agent)
# - [Rock Paper Scissors - Statistical Prediction](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-statistical-prediction)
# - [Rock Paper Scissors - Random Seed Search](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-random-seed-search)
# - [Rock Paper Scissors - RNG Statistics](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-rng-statistics)
# - [Rock Paper Scissors - XGBoost](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-xgboost)
# - [Rock Paper Scissors - Decision Tree](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-decision-tree)
