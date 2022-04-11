from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from epsilon_profile import EpsilonProfile
from controller.qagent import QAgent
import matplotlib.pyplot as plt
import numpy as np

def main():

    game = SpaceInvaders(display=True)
    print(game.screen_height, game.screen_width)
    
    #param apprentissage 

    n_episodes = 4
    gamma = 1
    alpha = 0.001
    eps_profile = EpsilonProfile(1.0, 1.0)
    
    ###Test sur l'apprentissage de l'agent, décommentez pour essayer le DL
    ##définition de l'agent
    controller = QAgent(game, eps_profile, gamma, alpha)

    scores = controller.learn(game, n_episodes)
    les_x=np.linspace(0, 200)
    print (controller.Q)
    plt.plot(les_x, scores)
    plt.show()

    ###Test sur le jeu tout seul, décommentez si besoin de tester get_state()
    """ controller = KeyboardController()
    state=game.reset()
    
    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        print(state)
        #sleep(0.001) """

if __name__ == '__main__' :
    main()
