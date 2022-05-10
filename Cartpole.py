#Cartpole problem on the open ai gym
#Philipp Renner

import numpy as np
import gym
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, InputLayer
from collections import deque

env = gym.make('CartPole-v0')

EPISODES = 2000
BATCH_SIZE = 64
DISCOUNT = 0.95
UPDATE_TARGET_EVERY = 5
STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n

class DQNAgents:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_memory = deque(maxlen = 5_000)
        self.gamma = 0.995
        #initial probability to choose a random action
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.model = self._build_model()
        self.target_model = self.model
        
        self.target_update_counter = 0
        print('Initialize the agent')
        
    def _build_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=self.state_size))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(self.action_size, activation = 'linear'))
        model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(lr = 0.001))
        return model
    
    #function wether an action should be random to explore or learned to maximize the benefit
    def action_choice(self, current_state):
        if np.random.random() > self.epsilon:
            action = np.argmax(self.get_qs(current_state))
        else:
            action = env.action_space.sample()
        
        return action
        


    def update_replay_memory(self, current_state, action, reward, next_state, done):
        self.replay_memory.append((current_state, action, reward, next_state, done))
        
        
    def train(self, terminal_state):
        # Sample from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        #Picks the current states from the randomly selected minibatch
        current_states = np.array([t[0] for t in minibatch])
        current_qs_list= self.model.predict(current_states) #gives the Q value for the policy network
        new_state = np.array([t[3] for t in minibatch])
        future_qs_list = self.target_model.predict(new_state)
        
        X = []
        Y = []
        
        # This loop will run 32 times (actually minibatch times)
        for index, (current_state, action, reward, next_state, done) in enumerate(minibatch):
            
            if not done:
                new_q = reward + DISCOUNT * np.max(future_qs_list[index])
            else:
                new_q = reward
                
            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            
            X.append(current_state)
            Y.append(current_qs)
        
        # Fitting the weights, i.e. reducing the loss using gradient descent
        self.model.fit(np.array(X), np.array(Y), batch_size = BATCH_SIZE, verbose = 0, shuffle = False)
        
       # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1
            
        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
            
def main():
    agent = DQNAgents(STATE_SIZE, ACTION_SIZE)

    for e in range(EPISODES):
    
        done = False
        current_state = env.reset()
        steps = 0 
        total_reward = 0
        while not done:
            if e>50:env.render()
            action = agent.action_choice(current_state)
            
            next_state, reward, done, _ = env.step(action)
            
            #calculating the new reaward the smaler the angle the bigger the reward
            reward = reward/abs(next_state[2]) * 0.03

            
            agent.update_replay_memory(current_state, action, reward, next_state, done)
        
            if len(agent.replay_memory) < BATCH_SIZE:
                pass
            else:
                agent.train(done)
            
            steps+=1    
            current_state = next_state
            total_reward += reward
            
            
        
        print(f"episode : {e}, steps {steps}, epsilon : {agent.epsilon}")
    
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
      
        
if __name__ == "__main__":
    main()
