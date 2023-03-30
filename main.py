import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs



#Step1 : adjusting the hyperparameters : hit and trial (avg reward for 1st day training(march9 = 890) )


class Hp():
    def __init__(self):
        self.nb_steps = 1000            # Number of training steps
        self.episode_lenght = 1000      # Maximum length of each episode
        self.learning_rate = 0.02       # Learning rate for updating the policy
        self.nb_directions = 16         # Number of perturbations used to estimate the gradient
        self.nb_best_directions = 16    # Number of best perturbations used to update the policy
        assert self.nb_best_directions <= self.nb_directions  # Check if nb_best_directions is less than or equal to nb_directions
        self.noise = 0.03               # Amount of noise added to the perturbations
        self.seed = 1                   # Random seed used to initialize the algorithm
        self.env_name = "HalfCheetahBulletEnv-v0"  # Name of the OpenAI Gym environment used for training





#------------------------------------------------>>

#normalise to scale and center the input data
#easier to recongise pattern later on 


class Normalizer():
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    #observe : to compute the running mean and variance of the input data
    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean)*(x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    #normalise : subtract the mean and divide by the standard deviation

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std



#------------------------------------------------>>
class Policy():

    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))

## Calculate the model output for and input 
#find the pertubation/change based on the direction 
# The direction argument specifies whether to add or subtract the delta from the weights

    def evaluate(self, input, delta=None, direction=None): #If direction is None, then the policy function is evaluated without any perturbations.
        if direction is None:
            return self.theta.dot(input)
        
        #If direction is "positive", then the policy function is evaluated with a positive perturbation.
        
        elif direction == "positive": 
            return (self.theta + hp.noise*delta).dot(input)

        else:
            return (self.theta - hp.noise*delta).dot(input) #If direction is "negative", then the policy function is evaluated with a negative perturbation.


    #delta is the noise vector used for positive and negative perturbations.
    def sample_deltas(self):
       return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]



    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)

        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg)*d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step #Each noise vector has the same shape as the theta parameter




def explore(env, normalizer, policy, direction=None, delta=None):
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done and num_plays < hp.episode_lenght:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards

#---------------------------------------------->>>>>


def train(env, policy, normalizer, hp):

    for step in range(hp.nb_steps):

        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions

        for k in range(hp.nb_directions):
            positive_rewards[k] = explore(
                env, normalizer, policy, direction="positive", delta=deltas[k])

        for k in range(hp.nb_directions):
            negative_rewards[k] = explore(
                env, normalizer, policy, direction="negative", delta=deltas[k])

        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()

        scores = {k: max(r_pos, r_neg)for k, (r_pos, r_neg)
                  in enumerate(zip(positive_rewards, negative_rewards))}

        order = sorted(scores.keys(), key=lambda x: scores[x])[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k])
                    for k in order]

        policy.update(rollouts, sigma_r)

        reward_evaluation = explore(env, normalizer, policy)
        print("Step: ", step, "Reward: ", reward_evaluation)


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


work_dir = mkdir('vid')
monitor_dir = mkdir(work_dir, 'monitor')

hp = Hp()
np.random.seed(hp.seed)
env = gym.make(hp.env_name)
env = wrappers.Monitor(env, monitor_dir, force=True)
nb_inputs = env.observation_space.shape[0]
nb_outputs = env.action_space.shape[0]
policy = Policy(nb_inputs, nb_outputs)
normalizer = Normalizer(nb_inputs)
train(env, policy, normalizer, hp)
