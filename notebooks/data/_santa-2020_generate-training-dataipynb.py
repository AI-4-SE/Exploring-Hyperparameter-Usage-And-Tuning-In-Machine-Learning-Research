#!/usr/bin/env python
# coding: utf-8

# # Collect training data from episodes
# 
# The next step beyond heuristics and careful mathematics is to train a machine learning model to choose which lever we should pull.  This notebook shows how to pull some basic training features from matches run locally using the kaggle_environments package.
# 

# In[ ]:


import pandas as pd
from kaggle_environments import make


# In[ ]:


# Parameters
rounds = 5
agent_dir = '../input/sample-submissions/'
agents = [
    'bayesian_ucb_xxx.py', 
    'simple_mab_1045.py',
    'thompson_xxx.py',
]


# In[ ]:


def log_training(result, n_machines):
    """Records training data from each machine, each agent, each round
    
    Generates a training dataset to support prediction of the current
    payout ratio for a given machine.
    
    Args:
       result ([[dict]]) - output from all rounds provided as output of 
                           env.run([agent1, agent2])
       n_machines (int) - number of machines
                           
    Returns:
       training_data (pd.DataFrame) - training data, including:
           "round_num"      : round number
           "machine_id"     : machine data applies to
           "agent_id"       : player data applies to (0 or 1)
           "n_pulls_self"   : number of pulls on this machine so far by agent_id
           "n_success_self" : number of rewards from this machine by agent_id
           "n_pulls_opp"    : number of pulls on this machine by the other player
           "payout"         : actual payout ratio for this machine
    
    """
    # Initialize machine and agent states
    machine_state = [{'n_pulls_0': 0, 'n_success_0': 0,
                      'n_pulls_1': 0, 'n_success_1': 0,
                      'payout': None}
                     for ii in range(n_machines)]
    agent_state = {'reward_0': 0, 'reward_1': 0, 'last_reward_0': 0,
                   'last_reward_1': 0}

    # Initialize training dataframe
    # - In the first round, store records for all n_machines
    # - In subsequent rounds, just store the two machines that updated
    training_data = pd.DataFrame(
            index=range(n_machines + 4 * (len(result) - 1)),
            columns=['round_num', 'machine_id', 'agent_id',
                     'n_pulls_self', 'n_success_self',
                     'n_pulls_opp', 'payout'])
    
    # Log training data from each round
    for round_num, res in enumerate(result):
        # Get current threshold values
        thresholds = res[0]['observation']['thresholds']

        # Update agent state
        for agent_ii in range(2):
            agent_state['last_reward_%i' % agent_ii] = (
                res[agent_ii]['reward']
                - agent_state['reward_%i' % agent_ii])
            agent_state['reward_%i' % agent_ii] = res[agent_ii]['reward']        

        # Update most recent machine state
        if res[0]['observation']['lastActions']:
            for agent_ii, r_obs in enumerate(res):
                action = r_obs['action']
                machine_state[action]['n_pulls_%i' % agent_ii] += 1
                machine_state[action]['n_success_%i' % agent_ii] += \
                    agent_state['last_reward_%i' % agent_ii]
                machine_state[action]['payout'] = thresholds[action]
        else:
            # Initialize machine states
            for mach_ii in range(n_machines):
                machine_state[mach_ii]['payout'] = thresholds[mach_ii]
            
        # Record training records
        # -- Each record includes:
        #       round_num, n_pulls_self, n_success_self, n_pulls_opp
        if res[0]['observation']['lastActions']:
            # Add results for most recent moves
            for agent_ii, r_obs in enumerate(res):
                action = r_obs['action']

                # Add row for agent who acted
                row_ii = n_machines + 4 * (round_num - 1) + 2 * agent_ii 
                training_data.at[row_ii, 'round_num'] = round_num
                training_data.at[row_ii, 'machine_id'] = action
                training_data.at[row_ii, 'agent_id'] = agent_ii
                training_data.at[row_ii, 'n_pulls_self'] = (
                    machine_state[action]['n_pulls_%i' % agent_ii])
                training_data.at[row_ii, 'n_success_self'] = (
                    machine_state[action]['n_success_%i' % agent_ii])
                training_data.at[row_ii, 'n_pulls_opp'] = (
                    machine_state[action]['n_pulls_%i' % (
                        (agent_ii + 1) % 2)])
                training_data.at[row_ii, 'payout'] = (
                    machine_state[action]['payout'] / 100)

                # Add row for other agent
                row_ii = n_machines + 4 * (round_num - 1) + 2 * agent_ii + 1
                other_agent = (agent_ii + 1) % 2
                training_data.at[row_ii, 'round_num'] = round_num
                training_data.at[row_ii, 'machine_id'] = action
                training_data.at[row_ii, 'agent_id'] = other_agent
                training_data.at[row_ii, 'n_pulls_self'] = (
                    machine_state[action]['n_pulls_%i' % other_agent])
                training_data.at[row_ii, 'n_success_self'] = (
                    machine_state[action]['n_success_%i' % other_agent])
                training_data.at[row_ii, 'n_pulls_opp'] = (
                    machine_state[action]['n_pulls_%i' % agent_ii])
                training_data.at[row_ii, 'payout'] = (
                    machine_state[action]['payout'] / 100)
                
        else:
            # Add initial data for all machines
            for action in range(n_machines):
                row_ii = action
                training_data.at[row_ii, 'round_num'] = round_num
                training_data.at[row_ii, 'machine_id'] = action
                training_data.at[row_ii, 'agent_id'] = -1
                training_data.at[row_ii, 'n_pulls_self'] = 0
                training_data.at[row_ii, 'n_success_self'] = 0
                training_data.at[row_ii, 'n_pulls_opp'] = 0
                training_data.at[row_ii, 'payout'] = (
                    machine_state[action]['payout'] / 100)
            
    return training_data


# # Generate example data

# In[ ]:


# Initilize environment
env = make("mab", debug=True)

# Create results
training_data = []
for r_num in range(rounds):
    print('***Round %i of %i***' % (r_num + 1, rounds))
    for ii, agent1 in enumerate(agents):
        for agent2 in agents[(ii+1):]:
            env.reset()
            result = env.run([agent_dir + agent1, agent_dir + agent2])
            training_data.append(log_training(
                result, len(result[0][0]['observation']['thresholds'])))
            print('agent 1 (%s): %i, agent 2 (%s): %i' % (
                agent1, result[-1][0]['reward'], agent2,
                result[-1][1]['reward']))

# Save training data
training_data = pd.concat(training_data, axis=0)
training_data.to_csv('/kaggle/working/training_data.csv', index=False, header=True)

