#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# A multiprocessing version of kernel https://www.kaggle.com/code/thomasdubail/kore2022-winrate-evaluation


# In[ ]:


## Version 3 - updates from this topic
## https://www.kaggle.com/competitions/kore-2022/discussion/324150


# In[ ]:


from kaggle_environments import make
env = make("kore_fleets", debug=True)
print(env.name, env.version)
from tqdm.contrib.concurrent import process_map
import numpy as np

agenta, agentb = "balanced", "balanced"

No_games_to_run = 100
show_result_per_game = True

def runs(i):
    env = make("kore_fleets")
    env.run([agenta, agentb])
    rewards = [x["reward"] for x in env.steps[-1]]
    scores1, scores2 = rewards[0], rewards[1]
    
    if scores1 > scores2:
        wins = 1
    else:
        wins = 0

    if show_result_per_game:
        what = 'Win' if wins==1 else 'Lost'
        print(f'Game no. #{i} : {what} with score {scores1:.0f} vs {scores2:.0f}')
    return scores1, scores2, wins


results = process_map(runs, range(No_games_to_run))
print(f' Win rate {100*np.array(results)[:,2].sum()/No_games_to_run}% with mean score {np.array(results)[:,0].mean():.0f} vs {np.array(results)[:,1].mean():.0f} ')

