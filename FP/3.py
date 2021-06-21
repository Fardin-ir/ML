import gym
import pandas as pd
import numpy as np
import time

custom_map = [
    'SFFFH',
    'FFHHF',
    'FFFFF',
    'HHFHF',
    'FFFFG'
]

custom_map_ch = pd.DataFrame(np.array([list(x) for x in custom_map]))

actions = {
    0: 'Left',
    1: 'Down',
    2: 'Right', 
    3: 'Up'
}


env = gym.make('FrozenLake-v0', desc=custom_map)
env.reset()

def value_iteration(env, max_iterations=10000, lmbda=0.85):
  start_time = time.time()
  policy = [0 for i in range(env.nS)]  
  stateValue = [0 for i in range(env.nS)]
  newStateValue = stateValue.copy()
  for i in range(max_iterations):
    for state in range(env.nS):
      action_values = []      
      for action in range(env.nA):
        state_value = 0
        for j in range(len(env.P[state][action])):
          prob, next_state, reward, done = env.P[state][action][j]
          state_action_value = prob * (reward + lmbda*stateValue[next_state])
          state_value += state_action_value
        action_values.append(state_value)      
        best_action = np.argmax(np.asarray(action_values))   
        policy[state] = best_action
        newStateValue[state] = action_values[best_action]  

    if sum(stateValue) - sum(newStateValue) == 0:
      break
    stateValue = newStateValue.copy()
  policy = np.asarray(policy).reshape((5,5))
  stateValue = pd.DataFrame(np.asarray(stateValue).reshape((5,5)))
  policy[np.where(custom_map_ch == 'H')] = -1
  policy[np.where(custom_map_ch == 'G')] = 111
  policy = pd.DataFrame(policy)
  print("--- %s seconds ---" % (time.time() - start_time))
  print("iter:",i)
  print(policy.replace(actions))
  print(stateValue)

def policy_iteration(env, max_iterations=10000, lmbda=0.85):
  start_time = time.time()
  policy = [0 for i in range(env.nS)]  
  stateValue = [0 for i in range(env.nS)]
  newStateValue = stateValue.copy()
  for i in range(max_iterations):
    for state in range(env.nS):
      action_values = []      
      action = policy[state]
      state_value = 0
      for j in range(len(env.P[state][action])):
        prob, next_state, reward, done = env.P[state][action][j]
        state_action_value = prob * (reward + lmbda*stateValue[next_state])
        state_value += state_action_value
      action_values.append(state_value)     
      best_action = np.argmax(np.asarray(action_values))   
      newStateValue[state] = action_values[best_action]  

    stateValue = newStateValue.copy()
    policy, policy_stable = get_policy(env,stateValue,policy.copy())
    if policy_stable:
      break
  policy = np.asarray(policy).reshape((5,5))
  stateValue = pd.DataFrame(np.asarray(stateValue).reshape((5,5)))
  policy[np.where(custom_map_ch == 'H')] = -1
  policy[np.where(custom_map_ch == 'G')] = 111
  policy = pd.DataFrame(policy)
  print("--- %s seconds ---" % (time.time() - start_time))
  print('iter:',i)
  print(policy.replace(actions))
  print(stateValue)

def get_policy(env,stateValue,old_policy,lmbda=0.85):
  policy_stable = True
  policy = [0 for i in range(env.nS)]
  for state in range(env.nS):
    action_values = []
    for action in range(env.nA):
      action_value = 0
      for i in range(len(env.P[state][action])):
        prob, next_state, r, _ = env.P[state][action][i]
        action_value += prob * (r + lmbda * stateValue[next_state])
      action_values.append(action_value)
    best_action = np.argmax(np.asarray(action_values))
    policy[state] = best_action
  policy_stable = old_policy==policy
  return policy,policy_stable


value_iteration(env)
