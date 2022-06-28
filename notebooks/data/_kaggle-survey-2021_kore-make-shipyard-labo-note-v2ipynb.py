#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install --upgrade kaggle_environments\n')


# In[ ]:


from kaggle_environments import make
import matplotlib.pylab as plt
env = make("kore_fleets",configuration={"randomSeed":42}, debug=True)
step6 = env.run(["../input/kore-make-shipyard-labo-note-v2/main8.py","../input/kore-make-shipyard-labo-note-v2/do_nothing.py"])
env.render(mode="ipython", width=700, height=700)


# pixyz  
# last update 2022 05 21  
# ゆっくりしていってね！  

# **霊夢:今回はコンペ理解のために、Shipyardについて検証していくぞ。**
# 
# **魔理沙:Shipyardは何個作るべきか、どこに作るべきかを検証していくぜ**
# 
# **Reimu: This time, I'm going to examine Shipyard to understand the competition.**
# 
# **Marisa: Let's verify how many Shipyards should be made and where to make them.**

# <img src="https://4.bp.blogspot.com/-uoVuBWIbdiA/WvQHqpx_YCI/AAAAAAABL8g/NiFZ6K71VBc_0_dcKb3_4nhnvFJ_JMNuACLcBGAs/s450/network_dennou_sekai_figure.png" width = 300>

# # Contents
# 
# * v1 [**Optimal number of shipyards**](#Optimal-number-of-shipyards)
# 
# * v2 [**share fleets**](#share-fleets)

# # Method

# **霊夢:色々なagentを、特定のマップで動かしてみて、以下の4つの変動をグラフにして比較してみるよ。**
# 
# * Kore:Shipyardsが持っている資源の数
# * Cargo:艦隊が持っている資源の数
# * Ships:全体が持っている船の数
# * All:これまでに獲得した資源の数 (Kore + Cargo + Ships×10 - 500)
# 
# **10,500 の値はそれぞれ、造船のコスト、ゲーム開始時に所持している資源の数の事です。**
# 
# **魔理沙:randomSeed = 42に設定して、Koreの配置は全検証において同じになるようにしたぜ**
# 
# **Reimu: Try running different agents on a specific map and compare the following four fluctuations in a graph.**
# 
# * Kore: Number of resources that Shipyards have
# * Cargo: The number of resources the fleet has
# * Ships: Number of ships the player has
# * All: Number of resources acquired so far (Kore + Cargo + Ships × 10 -500)
# 
# **Values of 10,500 are the cost of shipbuilding and the number of resources you have at the start of the game, respectively.**
# 
# **Marisa: I set randomSeed = 42 so that Kore's placement is the same for all validations**
# 
# <img src="https://3.bp.blogspot.com/-eZg7ny24Qkk/WkR91IX_MdI/AAAAAAABJVU/_E8oVVwiaRwUj2SN9h8hsIKpMR0yt3WgQCLcBGAs/s400/kabu_chart_man.png" width = 200>

# In[ ]:


def make_graph(steps):
    step_size = len(steps)
    Kore = [steps[i][0]["reward"] for i in range(step_size)]
    Cargo = [0]*step_size
    Ships = [0]*step_size
    All = [0]*step_size

    for i in range(step_size):
        for x in steps[i][0]['observation']["players"][0][2].values():
            Cargo[i] += x[1]
    for i in range(step_size):
        for x in steps[i][0]['observation']["players"][0][1].values():
            Ships[i] += x[1]
        for x in steps[i][0]['observation']["players"][0][2].values():
            Ships[i] += x[2]
    for i in range(step_size):
        All[i] = Kore[i]+Cargo[i]+(Ships[i]-50)*10 
            
    fig, ax = plt.subplots(figsize = (20, 10))
    plt.rcParams["font.size"] = 12
    plt.subplot(2,2,1)
    plt.plot(Kore)
    plt.title("Kore")
    plt.grid()
    plt.subplot(2,2,2)    
    plt.plot(Cargo)
    plt.title("Cargo")
    plt.grid()
    plt.subplot(2,2,3)
    plt.plot(Ships)
    plt.title('Ships')
    plt.grid()
    plt.subplot(2,2,4)
    plt.plot(All)
    plt.title('All')
    plt.grid()
    # plt.title(df_main.loc[0,columns[0]])


# In[ ]:


def vs_make_graph(step_list,agent_name):
    Kore = [] 
    Cargo = []
    Ships = []
    All = []
    for j,steps in enumerate(step_list):
        step_size = len(steps)
        Kore.append([steps[i][0]["reward"] for i in range(step_size)])
        Cargo.append([0]*step_size) 
        Ships.append([0]*step_size)
        All.append([0]*step_size)

        for i in range(step_size):
            for x in steps[i][0]['observation']["players"][0][2].values():
                Cargo[j][i] += x[1]
        for i in range(step_size):
            for x in steps[i][0]['observation']["players"][0][1].values():
                Ships[j][i] += x[1]
            for x in steps[i][0]['observation']["players"][0][2].values():
                Ships[j][i] += x[2]
        for i in range(step_size):
            All[j][i] = Kore[j][i]+Cargo[j][i]+(Ships[j][i]-50)*10 

    fig, ax = plt.subplots(figsize = (20, 30))
    plt.rcParams["font.size"] = 20
    plt.subplot(4,1,1)
    for i in range(len(step_list)):    
        plt.plot(Kore[i],label=agent_name[i])
    plt.title("Kore")
    plt.legend()
    plt.grid()
    plt.subplot(4,1,2)    
    for i in range(len(step_list)):    
        plt.plot(Cargo[i],label=agent_name[i])
    plt.title("Cargo")
    plt.legend()
    plt.grid()
    plt.subplot(4,1,3)
    for i in range(len(step_list)): 
        plt.plot(Ships[i],label=agent_name[i])
    plt.title('Ships')
    plt.legend()
    plt.grid()
    plt.subplot(4,1,4)
    for i in range(len(step_list)):    
        plt.plot(All[i],label=agent_name[i])
    plt.title('All')
    plt.legend()
    plt.grid()


# In[ ]:


get_ipython().run_cell_magic('writefile', 'do_nothing.py', '# First we will make a do_nothing player to observe the game board\ndef do_nothing():\n    pass\n')


# # Optimal number of shipyards

# **霊夢:まずは、造船所の数を変化させて、資源の回収率について実験していくよ。**
# 
# **Reimu: First, let's change the number of shipyards and experiment with resource recovery rates.**
# 
# <img src="https://1.bp.blogspot.com/-thxh6HTbVZc/XRWCr8TvLDI/AAAAAAABTbk/7xUQSctuwbMHM5nGLuAw7Fbm41bE5K7dgCLcBGAs/s450/space_uchusen_sentouki_bg.png" width = 200>

# ## No.1 1shipyards

# In[ ]:


get_ipython().run_cell_magic('writefile', 'four_minning_5.py', "from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef box_minning(agent,turn,param):\n    box_size = 5\n    period = 20\n    for k,shipyard in enumerate(agent.shipyards):\n        if not shipyard.next_action == None:\n            continue\n        if shipyard.ship_count >= 21 and turn % period == 10 and param.box_num[k] < 1:\n            param.box_num[k] = 1\n        if shipyard.ship_count >= 64 and turn % period == 10 and param.box_num[k] < 2:\n            param.box_num[k] = 2\n        action = None\n        \n        Dir = ['ENWS','WSEN']\n        r = [10,0]\n        add_ship = 0 if param.box_num[k] == 0 else (param.ship_num[k] - 32 * param.box_num[k])//(param.box_num[k]*box_size)\n        if param.shipyard_count < param.max_shipyard : add_ship = min(add_ship,15)\n        add_ship = max(add_ship,0)\n        for num in range(param.box_num[k]):\n            for i in range(box_size):\n                if turn % period == (2*i + r[num]) % period:\n                    if i == 0:\n                        flight_plan = ''\n                        for j in range(3):\n                            flight_plan += Dir[num][j] + str(box_size-1)\n                        flight_plan += Dir[num][3]\n                        action = ShipyardAction.launch_fleet_with_flight_plan(21 + add_ship, flight_plan)            \n                    elif i == box_size:\n                        flight_plan = Dir[num][1]\n                        action = ShipyardAction.launch_fleet_with_flight_plan(2 + add_ship, flight_plan)            \n                    elif i == box_size - 1:\n                        flight_plan = Dir[num][:2]\n                        action = ShipyardAction.launch_fleet_with_flight_plan(2 + add_ship, flight_plan)            \n                    else:\n                        flight_plan = Dir[num][0] + str(box_size - i - 1) + Dir[num][1]\n                        action = ShipyardAction.launch_fleet_with_flight_plan(3 + add_ship, flight_plan)            \n            \n        shipyard.next_action = action\n        \n    return\n")


# In[ ]:


get_ipython().run_cell_magic('writefile', 'spawn.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef spawn(agent,spawn_cost,kore_left,param):\n    for i,shipyard in enumerate(agent.shipyards):\n        if shipyard.next_action:\n            continue\n        if kore_left >= spawn_cost * shipyard.max_spawn:\n            action = ShipyardAction.spawn_ships(shipyard.max_spawn)\n            shipyard.next_action = action\n            kore_left -= spawn_cost * shipyard.max_spawn\n            param.ship_num[i] += shipyard.max_spawn\n        elif kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            shipyard.next_action = action\n            kore_left -= spawn_cost\n            param.ship_num[i] += 1\n    return\n')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'main1.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\nfrom four_minning_5 import box_minning\nfrom spawn import spawn\n\nclass param:\n    box_num = [0]\n    ship_num = [0]\n    shipyard_count = 1\n    max_shipyard = 1\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    box_minning(me,turn,param)\n    spawn(me,spawn_cost,kore_left,param)\n    return me.next_actions\n')


# In[ ]:


step1 = env.run(["/kaggle/working/main1.py","/kaggle/working/do_nothing.py"])
env.render(mode="ipython", width=1000, height=800)


# ### results

# In[ ]:


make_graph(step1)


# ## No.2 2shipyards

# In[ ]:


get_ipython().run_cell_magic('writefile', 'main2.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\nfrom four_minning_5 import box_minning\nfrom spawn import spawn\nfrom make_shipyard import make_shipyard\n\nclass param:\n    box_num = [0]\n    ship_num = [0]\n    shipyard_count = 1\n    max_shipyard = 2\n    \ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n    \n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    \n    box_minning(me,turn,param)\n    if param.max_shipyard > param.shipyard_count: make_shipyard(me,turn,param)\n    spawn(me,spawn_cost,kore_left,param)\n    return me.next_actions\n')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'make_shipyard.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\ndef make_shipyard(agent,turn,param):\n    period = 20\n    action = None\n    shipyard_posi = ["E5C"]\n    for i,shipyard in enumerate(agent.shipyards):\n        if not shipyard.id == \'0-1\':\n            continue\n        if shipyard.next_action:\n            continue            \n        if param.box_num[i] == 2 and shipyard.ship_count >= 82:\n            action = ShipyardAction.launch_fleet_with_flight_plan(82, shipyard_posi[param.shipyard_count - 1])\n            shipyard.next_action = action\n            param.shipyard_count += 1\n            param.box_num.append(0)\n            param.ship_num[i] -= 82\n            param.ship_num.append(32)\n            return\n        \n    return\n')


# In[ ]:


step2 = env.run(["/kaggle/working/main2.py","/kaggle/working/do_nothing.py"])
env.render(mode="ipython", width=1000, height=800)


# ### result

# In[ ]:


make_graph(step2)


# ## No.3 3shipyards

# In[ ]:


get_ipython().run_cell_magic('writefile', 'main3.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\nfrom four_minning_5 import box_minning\nfrom spawn import spawn\nfrom make_shipyard2 import make_shipyard\n\nclass param:\n    box_num = [0]\n    ship_num = [0]\n    shipyard_count = 1\n    max_shipyard = 3\n \n    \ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n    \n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    \n    box_minning(me,turn,param)\n    if param.max_shipyard > param.shipyard_count: make_shipyard(me,turn,param)\n    spawn(me,spawn_cost,kore_left,param)\n    return me.next_actions\n')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'make_shipyard2.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\ndef make_shipyard(agent,turn,param):\n    period = 20\n    action = None\n    shipyard_posi = ["E5C","S5C"]\n    for i,shipyard in enumerate(agent.shipyards):\n        if not shipyard.id == \'0-1\':\n            continue\n        if shipyard.next_action:\n            continue            \n        if param.box_num[i] == 2 and shipyard.ship_count >= 82:\n            action = ShipyardAction.launch_fleet_with_flight_plan(82, shipyard_posi[param.shipyard_count - 1])\n            shipyard.next_action = action\n            param.shipyard_count += 1\n            param.box_num.append(0)\n            param.ship_num[i] -= 82\n            param.ship_num.append(32)\n            return\n        \n    return\n')


# In[ ]:


step3 = env.run(["/kaggle/working/main3.py","/kaggle/working/do_nothing.py"])
env.render(mode="ipython", width=1000, height=800)


# ### result

# In[ ]:


make_graph(step3)


# ## 1shipyard vs 2shipyard vs 3shipyard

# In[ ]:


step_list = [step1,step2,step3]
agent_name = ["1stepyard","2shipyard","3shipyard"]
vs_make_graph(step_list,agent_name)


# **霊夢:どのステータスを比べても、造船所が多い方が多くなってるね。**
#     
# **魔理沙:造船所を作り始めれるのは約200ターン目あたりになってからだから、それまではどのパターンも変わらないな。**
# 
# **Reimu: No matter what status you compare, there are more shipyards.**
#     
# **Marisa: It's only around the 200th turn that we can start building a shipyard, so until then, no pattern has changed.**

# # share fleets

# **霊夢:次は、造船所同士で、船を共有させるagentを作成してみるよ。**
# 
# **Reimu: Next, let's create an agent that allows shipyards to share ships.**

# ## No.4 2shipyard (share fleets)

# In[ ]:


get_ipython().run_cell_magic('writefile', 'four_minning_5_share.py', "from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef box_minning(agent,turn,param):\n    box_size = 5\n    period = 20\n    for k,shipyard in enumerate(agent.shipyards):\n        if not shipyard.next_action == None:\n            continue\n        if shipyard.ship_count >= 21 and turn % period == 10 and param.box_num[k] < 1:\n            param.box_num[k] = 1\n        if shipyard.ship_count >= 64 and turn % period == 10 and param.box_num[k] < 2:\n            param.box_num[k] = 2\n        if shipyard.ship_count >= 32 and turn % period == 0 and param.box_num[k] < 2:\n            param.box_num[k] = 2\n            \n        action = None\n        \n        Dir = ['ENWS','WSEN']\n        r = [10,0]\n        add_ship = 0 if param.box_num[k] == 0 else (param.ship_num[k] - 32 * param.box_num[k])//(param.box_num[k]*box_size)\n        if param.shipyard_count < param.max_shipyard : add_ship = min(add_ship,15)\n        add_ship = max(add_ship,0)\n        for num in range(param.box_num[k]):\n            for i in range(box_size):\n                if turn % period == (2*i + r[num]) % period:\n                    if i == 0:\n                        flight_plan = ''\n                        for j in range(3):\n                            flight_plan += Dir[num][j] + str(box_size-1)\n                        flight_plan += Dir[num][3]\n                        if not param.share_fleets[k][num] == -1 and param.share_fleets[k][num] < param.shipyard_count:\n                            param.ship_num[k] -= 21 + add_ship\n                            param.ship_num[param.share_fleets[k][num]] += 21 + add_ship\n                        action = ShipyardAction.launch_fleet_with_flight_plan(21 + add_ship, flight_plan)            \n                    elif i == box_size:\n                        flight_plan = Dir[num][1]\n                        action = ShipyardAction.launch_fleet_with_flight_plan(2 + add_ship, flight_plan)            \n                    elif i == box_size - 1:\n                        flight_plan = Dir[num][:2]\n                        action = ShipyardAction.launch_fleet_with_flight_plan(2 + add_ship, flight_plan)            \n                    else:\n                        flight_plan = Dir[num][0] + str(box_size - i - 1) + Dir[num][1]\n                        action = ShipyardAction.launch_fleet_with_flight_plan(3 + add_ship, flight_plan)            \n            \n        shipyard.next_action = action\n        \n    return\n")


# In[ ]:


get_ipython().run_cell_magic('writefile', 'main5.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\nfrom four_minning_5_share import box_minning\nfrom spawn import spawn\nfrom make_shipyard3 import make_shipyard\n\nclass param:\n    box_num = [0]\n    ship_num = [0]\n    shipyard_count = 1\n    max_shipyard = 2\n    share_fleets = [[-1,1],[0,-1]]\n    \ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n    \n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    \n    box_minning(me,turn,param)\n    if param.max_shipyard > param.shipyard_count: make_shipyard(me,turn,param)\n    spawn(me,spawn_cost,kore_left,param)\n    return me.next_actions\n')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'make_shipyard3.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\ndef make_shipyard(agent,turn,param):\n    period = 20\n    action = None\n    shipyard_posi = ["W4S4C"]\n    for i,shipyard in enumerate(agent.shipyards):\n        if not shipyard.id == \'0-1\':\n            continue\n        if shipyard.next_action:\n            continue            \n        if param.box_num[i] == 2 and shipyard.ship_count >= 82:\n            action = ShipyardAction.launch_fleet_with_flight_plan(82, shipyard_posi[param.shipyard_count - 1])\n            shipyard.next_action = action\n            param.shipyard_count += 1\n            param.box_num.append(0)\n            param.ship_num[i] -= 82\n            param.ship_num.append(32)\n            return\n        \n    return\n')


# In[ ]:


step4 = env.run(["/kaggle/working/main5.py","/kaggle/working/do_nothing.py"])
env.render(mode="ipython", width=1000, height=800)


# ### result

# In[ ]:


make_graph(step4)


# ## no_share vs share

# In[ ]:


step_list = [step2,step4]
agent_name = ["no_share","share"]
vs_make_graph(step_list,agent_name)


# **霊夢:あんまり結果が良くなってないぞ？**
# 
# **魔理沙:共有のやり方が良くないんじゃないか？片方が船を造船し続けて、片方が出航し続けるように役割分担させないと**
# 
# **Reimu: The results aren't getting much better, right?**
# 
# **Marisa: Isn't the sharing method bad? I have to divide the roles so that one keeps building the ship and the other keeps sailing**

# ## No.5 2shipyard (share fleets) ver.2

# In[ ]:


get_ipython().run_cell_magic('writefile', 'four_minning_5_share2.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef box_minning(agent,turn,param):\n    box_size = 5\n    period = 20\n    for k,shipyard in enumerate(agent.shipyards):\n        if not shipyard.next_action == None:\n            continue\n        if shipyard.ship_count >= 21 and turn % period == 10 and param.box_num[k] < 1:\n            param.box_num[k] = 1\n        if shipyard.ship_count >= 64 and turn % period == 10 and param.box_num[k] < 2:\n            param.box_num[k] = 2\n        if shipyard.ship_count >= 96 and turn % period == 10 and param.box_num[k] < 3:\n            param.box_num[k] = 3\n        if shipyard.ship_count >= 128 and turn % period == 10 and param.box_num[k] < 4:\n            param.box_num[k] = 4\n        \n        if not param.parent[k] == -1:\n            param.box_num[k] = 0\n        \n        action = None\n        \n        Dir = [\'ENWS\',\'WSEN\',\'ESWN\',"WNES"]\n        r = [10,11,0,1]\n        add_ship = 0 if param.box_num[k] == 0 else (param.ship_num[k] - 32 * param.box_num[k])//(param.box_num[k]*box_size)\n        if param.shipyard_count < param.max_shipyard : add_ship = min(add_ship,15)\n        add_ship = max(add_ship,0)\n        for num in range(param.box_num[k]):\n            for i in range(box_size):\n                if turn % period == (2*i + r[num]) % period:\n                    if i == 0:\n                        flight_plan = \'\'\n                        for j in range(3):\n                            flight_plan += Dir[num][j] + str(box_size-1)\n                        flight_plan += Dir[num][3]\n                        action = ShipyardAction.launch_fleet_with_flight_plan(min(shipyard.ship_count,21 + add_ship), flight_plan)            \n                    elif i == box_size:\n                        flight_plan = Dir[num][1]\n                        action = ShipyardAction.launch_fleet_with_flight_plan(2 + add_ship, flight_plan)            \n                    elif i == box_size - 1:\n                        flight_plan = Dir[num][:2]\n                        action = ShipyardAction.launch_fleet_with_flight_plan(2 + add_ship, flight_plan)            \n                    else:\n                        flight_plan = Dir[num][0] + str(box_size - i - 1) + Dir[num][1]\n                        action = ShipyardAction.launch_fleet_with_flight_plan(3 + add_ship, flight_plan)            \n            \n        shipyard.next_action = action\n        \n    return\n')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'supply.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef supply(agent,turn,param):\n    box_size = 5\n    period = 20\n    for k,shipyard in enumerate(agent.shipyards):\n        if param.parent[k] == -1:\n            continue\n        if not shipyard.next_action == None:\n            continue\n        num = param.parent[k]\n        Dir = [\'EN\',\'WS\',\'ES\',"WN"]\n        r = [1,0,11,10]\n        action = None\n        for ID,i in param.child[num].items():\n            if turn % period == r[i]:\n                flight_plan = Dir[i][0] + str(box_size - 1) + Dir[i][1]\n                if shipyard.ship_count > 2:\n                    if i%2 == 0:\n                        action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count, flight_plan)\n                        param.ship_num[ID] += param.ship_num[k]\n                        param.ship_num[k] = 0\n                    else:\n                        action = ShipyardAction.launch_fleet_with_flight_plan(max(3,shipyard.ship_count-param.ship_num[k]//2), flight_plan)\n                        param.ship_num[ID] += max(3,param.ship_num[k]//2)\n                        param.ship_num[k] -= max(3,param.ship_num[k]//2)\n        if not action:\n            param.ship_num[k] = shipyard.ship_count\n        shipyard.next_action = action        \n    return        \n')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'main6.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\nfrom four_minning_5_share2 import box_minning\nfrom spawn import spawn\nfrom make_shipyard4 import make_shipyard\nfrom supply import supply\nclass param:\n    box_num = [0]\n    ship_num = [0]\n    shipyard_count = 1\n    max_shipyard = 2\n    parent = [-1]\n    child = []\n    \ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n    \n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    \n    if param.max_shipyard == param.shipyard_count: supply(me,turn,param)\n    box_minning(me,turn,param)\n    if param.max_shipyard > param.shipyard_count: make_shipyard(me,turn,param)\n    spawn(me,spawn_cost,kore_left,param)\n    return me.next_actions\n')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'make_shipyard4.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\ndef make_shipyard(agent,turn,param):\n    period = 20\n    action = None\n    shipyard_posi = ["W4S4C"]\n    for i,shipyard in enumerate(agent.shipyards):\n        if not shipyard.id == \'0-1\':\n            continue\n        if shipyard.next_action:\n            continue            \n        if param.box_num[i] == 2 and shipyard.ship_count >= 50:\n            action = ShipyardAction.launch_fleet_with_flight_plan(50, shipyard_posi[param.shipyard_count - 1])\n            shipyard.next_action = action\n            param.shipyard_count += 1\n            param.box_num.append(0)\n            param.ship_num[i] -= 50\n            param.ship_num.append(0)\n            return\n        \n    return\n')


# In[ ]:


step5 = env.run(["/kaggle/working/main6.py","/kaggle/working/do_nothing.py"])
env.render(mode="ipython", width=1000, height=800)


# ### result

# In[ ]:


make_graph(step5)


# ## no_share vs share ver.2

# In[ ]:


step_list = [step2,step5]
agent_name = ["no_share","share_ver2"]
vs_make_graph(step_list,agent_name)


# ## No.6 3shipyard(share fleets)

# In[ ]:


get_ipython().run_cell_magic('writefile', 'main8.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\nfrom four_minning_5_share2 import box_minning\nfrom spawn import spawn\nfrom make_shipyard6 import make_shipyard\nfrom supply import supply\nclass param:\n    box_num = [0]\n    ship_num = [0]\n    shipyard_count = 1\n    max_shipyard = 3\n    parent = [-1]\n    parent_num = 0\n    child = []\n    \ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n    \n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    \n    if param.shipyard_count > 1:supply(me,turn,param)\n    if param.max_shipyard > param.shipyard_count: make_shipyard(me,turn,param)\n    box_minning(me,turn,param)\n    spawn(me,spawn_cost,kore_left,param)\n    return me.next_actions\n')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'make_shipyard6.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\ndef make_shipyard(agent,turn,param):\n    period = 20\n    action = None\n    shipyard_posi = ["E4N4C","W4S4C"]\n    r = [0,1,10,11]\n    for i,shipyard in enumerate(agent.shipyards):\n        if not shipyard.id == \'0-1\':\n            continue\n        if shipyard.next_action:\n            continue            \n        if not turn % period in r and shipyard.ship_count >= 82:\n            action = ShipyardAction.launch_fleet_with_flight_plan(82, shipyard_posi[param.shipyard_count - 1])\n            shipyard.next_action = action\n            param.box_num.append(0)\n            param.parent.append(-1)\n            param.ship_num[i] -= 82\n            param.ship_num.append(32)\n            if param.parent[i] == -1:\n                param.parent[i] = param.parent_num\n                param.parent_num += 1\n                param.child.append({})\n            param.child[param.parent[i]][param.shipyard_count] = param.shipyard_count - 1\n            param.shipyard_count += 1\n            return\n        \n    return\n')


# In[ ]:


step6 = env.run(["/kaggle/working/main8.py","/kaggle/working/do_nothing.py"])
env.render(mode="ipython", width=1000, height=800)


# ### result

# In[ ]:


make_graph(step6)


# ## no-share vs share ver.3

# In[ ]:


step_list = [step3,step6]
agent_name = ["no-share","share"]
vs_make_graph(step_list,agent_name)


# **霊夢:船の数はshareの方が少ないけど、合計の資源の取得量は対して変わらないね**
# 
# **魔理沙:shipyardの数をn個とすると、boxの数は、no_shareは2n,shareは4(n-1)になるからshipyardが増えるほど、Koreの回収が早くなりそうだね。**
# 
# **Reimu: The number of ships is smaller in share, but the total amount of resources acquired is the same**
# 
# **Marisa: Assuming that the number of shipyards is n, the number of boxes is 2n for no_share and 4 (n-1) for share, so the more shipyards there are, the faster Kore will be collected.**
