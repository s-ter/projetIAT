import numpy as np
from game.SpaceInvaders import SpaceInvaders
from epsilon_profile import EpsilonProfile
from time import sleep

class QAgent ():
    
    def __init__(self, game, eps_profile: EpsilonProfile, gamma: float, alpha: float, pos = True):
        
        self.pos = pos
        
        if pos:
            quantif, pos, feu, na = 20, 2, 2, 4 #On a 20 plages de quantifications différentes pour la distance, deux états possibles pour la balle, deux états possibles pour la position relatives et 4 actions possibles 
            self.Q = np.zeros([quantif, pos, feu, na])
        else:
            quantif, feu, na = 20, 2, 4 #On a 20 plages de quantifications différentes pour la distance, deux états possibles pour la balle, deux états possibles pour la position relatives et 4 actions possibles 
            self.Q = np.zeros([quantif, feu, na])
        
        self.game = game
        self.na = na
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_delay = 0.85
        
        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

    def learn(self, env, n_episodes): 
        scores = []
        # Execute N episodes (parties)
        for episode in range(n_episodes):
            state = env.reset()
            game_over = False
            score = 0
            print("Partie n°"+str(episode))
            while not game_over and score < 1000 :
                # Selectionne une action 
                action = self.select_action(state)
                # Echantillonne l'état suivant et la récompense
                next_state, reward, game_over, score = env.step(action)
                
                if not game_over:
                    # Mets à jour la fonction de valeur Q
                    self.updateQ(state, action, reward, next_state)
                    state = next_state
                else :
                    self.updateQ(state, action, 0, state)
                    scores.append(score)
                if score > 999:
                    scores.append(score)
                
            self.epsilon *= self.epsilon_delay


        return (scores)
  

    def updateQ(self, state, action, reward, next_state):
        ##print(state)
        #print(action)
        if not self.pos :
            state = (state[0], state[2])    
        self.Q[state][action] = (1. - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))
          
            
    def select_action(self, state : "Tuple[int, int]"):
        ###On choisit une action entre une action d'exploration ou une action d'optimale selon un param. externe epsilon
        #print(self.epsilon)
        if np.random.rand()<self.epsilon:
            a = np.random.randint(self.na)      # random action
        else:
            a = self.select_greedy_action(state)
        return a

    def select_greedy_action(self, state : "Tuple[int, int]"):
        ###On sélectionne la meilleure action possible
        mx = np.max(self.Q[state])
        return np.random.choice(np.where(self.Q[state] == mx)[0])