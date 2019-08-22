from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt 
import torch
from collections import deque

from ddpg_agent import Agent

env = UnityEnvironment(file_name = 'Reacher.app')
agent = Agent(state_size = 33, action_size = 4, random_seed = 0)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode = True)[brain_name]     # reset the environment  
action_size = brain.vector_action_space_size

scores = [] 													 # initialize the score 
scores_window = deque(maxlen = 100) 
                          
def ddpg(n_episodes = 1000, max_t = 1000):
	
	for i_episode in range(1, n_episodes + 1):
	   	env_info = env.reset(train_mode = True)[brain_name]     
	   	state = env_info.vector_observations[0]					# get the current state 
	   	agent.reset()
	   	score = 0

	   	for t in range(max_t):
	   		action = agent.act(state)							# select an action
		   	action = np.clip(action, -1, 1)                  	# all actions between -1 and 1
		   	env_info = env.step(action)[brain_name]           	# send all actions to tne environmentç
		   	next_state = env_info.vector_observations[0]        # get next state (for each agent)
		   	reward = env_info.rewards[0]                        # get reward (for each agent)
		   	done = env_info.local_done[0]    					# see if episode finished
		   	agent.step(state, action, reward, next_state, done)
		   	state = next_state									# roll over states to next time step								
		   	score += reward										# update the score 
		   	
		   	if done:											# exit loop if episode finished
		   		break

	   	scores_window.append(score)
	   	scores.append(score)
	   	print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end = "")

	   	if i_episode % 10:
	   		torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
	   		torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

   		if np.mean(scores_window) >= 30.0:
   			print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format
				(i_episode - 100, np.mean(scores_window)))
   			torch.save(agent.actor_local.state_dict(), 'checkpoint_finished.pth')
   			break

	return scores

if __name__ == '__main__':

	ddpg()

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(np.arange(len(scores)), scores)
	plt.title('Progress of the agent over the episodes')
	plt.ylabel('Score')
	plt.xlabel('Episode #')
	plt.show()
	   	    