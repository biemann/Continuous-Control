from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch

from ddpg_agent import Agent


env = UnityEnvironment(file_name = "Reacher.app")

agent = Agent(state_size = 33, action_size = 4, random_seed = 0)
agent.actor_local.load_state_dict(torch.load('bin/checkpoint_finished.pth', map_location = lambda storage, loc: storage))

brain_name = env.brain_names[0]
brain = env.brains[brain_name]


env_info = env.reset(train_mode = False)[brain_name]
state = env_info.vector_observations[0]
score = 0

while True:         
	action = agent.act(state)							# select an action
	action = np.clip(action, -1, 1)                  	# all actions between -1 and 1
	env_info = env.step(action)[brain_name]           	# send all actions to tne environmentç
	next_state = env_info.vector_observations[0]        # get next state (for each agent)
	reward = env_info.rewards[0]                        # get reward (for each agent)
	done = env_info.local_done[0]    					# see if episode finished
	state = next_state									# roll over states to next time step								
	score += reward										# update the score 
	if done:											# exit loop if episode finished
		break
print(score)           
env.close()