import numpy as np
from connection import connect
from connection import get_state_reward

s = connect(2037)
n_actions = 3
n_states = 24

#tabela que guarda as recompensas de cada estado e ação
Q_table = np.zeros((n_states*4, n_actions))

#número de vezes que ele vai chegar ao final
n_episodes = 100

#número de tentativas que ele tem a cada episode de chegar ao final
max_iter_episode = 100

#probabilidade de ele explorar algo novo e não se atentar necessariamente ao melhor valor conhecido
exploration_prob = 1.0

#uma vez que ele tenha ja conseguido chegar ao final uma vez, a necessidade de explorar diminui
exploration_decreasing_decay = 0.1

#teoricamente existe um minimo de exploração que precisa ter
min_exploration_prob = 0.1

#discount factor
gamma = 0.7

#learning rate
alpha = 0.1

#quanto custou o episode
total_rewards_episode = []

for i in range(n_episodes):
    current_state = "0b00000000"
    done = False
    
    total_episode_reward = 0
    
    for j in range(max_iter_episode):
        action_names = ["jump", "left", "right"]
        if np.random.uniform(0, 1) < exploration_prob:
            action = np.random.choice(action_names)
            print("random")
        else:
            action = action_names[np.argmax(Q_table[int(current_state, 2), :])]
            print("consciente")
        next_state, reward = get_state_reward(s, action)
        
        if reward == 300:
            done = True
        
        Q_table[int(current_state,2),action_names.index(action)] = (1-alpha) * Q_table[int(current_state,2),action_names.index(action)] + alpha*(reward + gamma*max(Q_table[int(next_state,2),:]))
        total_episode_reward = total_episode_reward + reward
        
        if done:
            break
        current_state = next_state
        #print(Q_table)
        print(i,j)
        print(exploration_prob)
    exploration_prob = max(min_exploration_prob, np.exp(-exploration_decreasing_decay*i))
    total_rewards_episode.append(total_episode_reward)

