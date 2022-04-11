import numpy as np
from game.SpaceInvaders import SpaceInvaders
from epsilon_profile import EpsilonProfile
from time import sleep

class QAgent ():
    
    def __init__(self, game, eps_profile: EpsilonProfile, gamma: float, alpha: float):
        
        quantif, feu, na = 20, 2, 4 #On a 20 plages de quantifications différentes, deux états possibles pour la balle et 4 actions possibles 
        self.Q = np.zeros([20, 2, na])
        
        self.game = game
        self.na = na
        
        self.alpha = alpha
        self.gamma = gamma
        
        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

    def learn(self, env, n_episodes, max_steps): 
        n_steps = np.zeros(n_episodes) + max_steps
        # Execute N episodes 
        for episode in range(n_episodes):
            state = env.reset()
            in_game = True
            while in_game:
                # Selectionne une action 
                action = self.select_action(state)
                # Echantillonne l'état suivant et la récompense
                next_state, reward, in_game = env.step(action)
                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state)
                state = next_state
            
            n_steps[episode] = step + 1  

    def updateQ(self, state, action, reward, next_state):
        #print(state)
        #print(next_state)
        self.Q[state][action] = (1. - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))
    
    def select_action(self, state : "Tuple[int, int]"):
        ###On choisit une action entre une action d'exploration ou une action d'optimale selon un param. externe epsilon
        
        if np.random.rand()<self.epsilon:
            a = np.random.randint(self.na)      # random action
        else:
            a = self.select_greedy_action(state)
        return a

    def select_greedy_action(self, state : "Tuple[int, int]"):
        ###On sélectionne la meilleure action possible
        mx = np.max(self.Q[state])
        return np.random.choice(np.where(self.Q[state] == mx)[0])