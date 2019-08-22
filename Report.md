## The DDPG algorithm

To solve the task, we use an implementation the Deep Deterministic Policy Gradient (DDPG) algorithm of the following paper: https://arxiv.org/pdf/1509.02971.pdf. The code is heavily inspired by the example used to solve the OpenAI-gym Pendulum task. Most of our work here was to adapt this algorithm to our Unity environment and especially modifying the hyperparameters to make the agent learn something.

The idea of the DDPG algorithm is an improvement of the Q-learning algorithm (see our previous Navigation project). The basic ideas are the same, as it also uses a replay buffer to reuse past experiments (to avoid a heavy correleation between subsequent states), as well as two parallel networks : the target and local network that do not update simultaneously in order to combat divergence of the algorithm. However, Q-learning is a value-based method and is not well-suited for continuous tasks (such as this one) and discrete tasks with high-dimensional action spaces. We also use soft updates, where the weights of the local network are slowly mixed into the target network.

Thie idea is to use policy-based method instead, using a critic network. The actor network approximates the optimal policy deterministically. The critic network is evaluating the actions taken by the actor network and tries to predict the reward of the chosen actions. The actor policy is updated using the sampled policy gradient.

In addition, in order to encourage exploration, they also implemented the Ornstein-Uhlenbeck noise. For more details, we refer to the original paper or to understand how it is implemented to the code.

## Implementation of the Q-learning algorithm

The `ddpg_agent.py`class is an implementation of the DDPG algorithm following the DeepMind paper as exposed in this nanodegree. The only slight modification we did was an implementation of a learning rate scheduler in order to help solving the task faster. 

The `model.py`is the neural network the DDPG algorithm uses to make the agent and the critic learn. The architectures will be described in the following section.

The `train.py`is the main function of our project. It adapts the DDPG algorithm to this particular environment. The code follows the notebook https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continuous-control/Continuous_Control.ipynb. Note that we slightly simplified the code, because this example used a version with 20 agents, whereas we only trained one agent.

## Network architecture and Hyperparameters

In contrast to the paper or the example, we used two different architectures for the actor and critc network. The actor network consists of 3 layers of 128 neurons each, followed by the tanh activation function (in order to have values between -1 and 1). The critc network is built with 3 layers of 64 neurons each. We use the selu activation function, instead of the more standard relu for both networks. In addition, we use batch normalisation only for the actor network (using it for both significantly hurt the training). 

We expermiented with the tau parameter (for soft updates), the gamma parameter (reward discount) and sigma (for data exploration). We finally settled for tau = 5e-4, gamma = 0.9 and sigma = 0.12, all three being lower than the original parameters. For the learning rate, we started for both networks with 5e-4 and divided it by 2 every 100 episodes, using a learning rate scheduler. Note that one episode consists of 1000 timesteps in our case, which is larger than for the Pendelum example. When lowering this parameter, the task will be solved in less episodes, but the same amount of time.

## Results

Initally, we had quite some problems to make the agent learn, concluding it is better to begin with a simple architecture and only implementing methods, such as Batch Normalisation once it was learning something. 

We were able to solve the task in 80 episodes (it took 180 episodes to get an average of 30.0 over the last 100 episodes). However, the algorithm is unstable and with the same parameters, we may need far more episodes. However, we expect that the task will be solved in less than 300 episodes.

We have here a graph, showing the learning of our best result. 

![Solved][image1]

We note that it reached the desired score of 30 consistently after 130 episodes and was being closed to 40 at the end of the training. The trend is quite similar to what we saw with the Q-learning algorithm : it has quite some trouble to learn anything at the beginning, learns fast in the middle and reaches a plateau at the end, because it learned the task (the maximum score should be around 40).

![Gif][image2]

We added a gif showing a trained agent. This agent reached a score of 38.0 and follows the ball pretty well during the whole episode.

## Future Work

In future, we could compare this method to other policy based methods, such as A2C, taking advantage of the parallelisation of different agents. We can also try this algorithm on more challenging environments, such as the crawler environment. This will require to optimise the architecture even more carefully to make the agent learn something. We would like to investigate such examples in future.
