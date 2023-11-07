import numpy as np 
import matplotlib.pyplot as plt 

# Define Action class 
class Bandit: 
    def __init__(self, m): 
        self.m = m 
        self.mean = 0
        self.N = 0

    # Random action 
    def choose(self): 
        return np.random.randn() + self.m 

    # Update the action-value estimate 
    def update(self, x): 
        self.N += 1
        self.mean = (1 - 1.0 / self.N)*self.mean + 1.0 / self.N * x 


def greedy(m1, m2, m3, eps, N): 
        
    actions = [Bandit(m1), Bandit(m2), Bandit(m3)] 

    data = np.empty(N) 
        
    for i in range(N): 
        if np.random.random() < eps: 
            j = np.random.choice(3) 
        else: 
            j = np.argmax([a.mean for a in actions]) 
        x = actions[j].choose() 
        actions[j].update(x) 

        # for the plot 
        data[i] = x 
    cumulative_average = np.cumsum(data) / (np.arange(N) +1) 

    # Sanity check
    for a in actions: 
        print(a.mean) 

    return cumulative_average 



# Formatting: mean of actions 1,2,3, epsilon, N 
eps_1 = greedy(1.0, 2.0, 3.0, 0.1, 100000) 
eps_05 = greedy(1.0, 2.0, 3.0, 0.05, 100000) 
eps_01 = greedy(1.0, 2.0, 3.0, 0.01, 100000) 
# Plot all experiments
plt.figure(figsize = (10, 6))
plt.plot(eps_01, label ='epilons = 0.01') 
plt.plot(eps_05, label ='epsilon = 0.05') 
plt.plot(eps_1, label ='epsilon = 0.1') 
plt.legend() 
plt.xscale('log') 
plt.show() 
