from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from epsilon_profile import EpsilonProfile
from controller.qagent import QAgent
import matplotlib.pyplot as plt
import numpy as np

global path_to_save
path_to_save = "/home/juju/Devoirs/INSA/4A/Semestre 2/Intelligence_Artificielle_telecommunication_IAT/Simulations_fig/"

def simu_simple(n_episodes, gamma, alpha, epsilon_value):
    """Simulation et retourne les courbes d'apprentissage"""
    game = SpaceInvaders(display=True)
    eps_profile = EpsilonProfile(epsilon_value, epsilon_value)
    controller = QAgent(game, eps_profile, gamma, alpha, pos=False)

    scores = controller.learn(game, n_episodes)
    les_x=[i+1 for i in range(n_episodes)]
    plt.plot(les_x, scores, 'xb-')
    plt.xlabel("Nb de parties")
    plt.ylabel("Score")
    plt.title("Évolution des scores sur "+str(n_episodes)+"\n(gamma ="+str(gamma)+", alpha="+str(alpha)+", espilon_value="+str(epsilon_value)+")")
    plt.savefig(path_to_save+"simu_simple_"+str(n_episodes)+"_"+str(gamma)+"_"+str(alpha)+"_"+str(epsilon_value)+".png")
    plt.close()
    
def simu_simple_with_pos_state(n_episodes, gamma, alpha, epsilon_value):
    """Simulation et retourne les courbes d'apprentissage"""
    game = SpaceInvaders(display=True)
    eps_profile = EpsilonProfile(epsilon_value, epsilon_value)
    controller = QAgent(game, eps_profile, gamma, alpha)

    scores = controller.learn(game, n_episodes)
    les_x=[i+1 for i in range(n_episodes)]
    plt.plot(les_x, scores, 'xb-')
    plt.xlabel("Nb de parties")
    plt.ylabel("Score")
    plt.title("Évolution des scores sur "+str(n_episodes)+"\n(gamma ="+str(gamma)+", alpha="+str(alpha)+", espilon_value="+str(epsilon_value)+")")
    plt.savefig(path_to_save+"simu_simple_with_pos_state"+str(n_episodes)+"_"+str(gamma)+"_"+str(alpha)+"_"+str(epsilon_value)+".png")
    plt.close()
    
def simu_moyenne(n_episodes, gamma, alpha, epsilon_value):
    """Simulation et retourne les courbes d'apprentissage"""
    game = SpaceInvaders(display=True)
    eps_profile = EpsilonProfile(epsilon_value, epsilon_value)
    controller = QAgent(game, eps_profile, gamma, alpha)

    scores = controller.learn(game, n_episodes)
    scores_moy = [(sum(scores[i:i+10])/10) for i in range(0, n_episodes, 10)]
    les_x=[(i+1)*10 for i in range(n_episodes//10)]
    print(len(les_x))
    plt.plot(les_x, scores_moy, 'xb-')
    plt.xlabel("Nb de parties")
    plt.ylabel("Score")
    plt.title("Évolution de la moyenne du scores sur 10 parties\n(gamma ="+str(gamma)+", alpha="+str(alpha)+", espilon_value="+str(epsilon_value)+")")
    plt.savefig(path_to_save+"simu_moyenne_"+str(n_episodes)+"_"+str(gamma)+"_"+str(alpha)+"_"+str(epsilon_value)+".png")
    plt.close()

def simu_greedy(n_learn, n_test, tot, gamma, alpha, epsilon_value):
    """Simulation et retourne les courbes d'apprentissage"""
    scores = []
    game = SpaceInvaders(display=True)
    eps_profile = EpsilonProfile(epsilon_value, epsilon_value)
    controller = QAgent(game, eps_profile, gamma, alpha)
    
    for i in range(tot):
        controller.eps_profile = EpsilonProfile(epsilon_value, epsilon_value)
        controller.learn(game, n_learn)
        
        #on évalue l'efficacité de l'alien quand celui-ci choisi toujours la meilleure action sur 10 parties
        eps_profile = EpsilonProfile(0, 0)
        controller.eps_profile = eps_profile
        scores+=controller.learn(game, n_test)
        
    les_x=[i+1 for i in range(len(scores))]
    plt.plot(les_x, scores, 'xb-')
    plt.xlabel("Nb de parties")
    plt.ylabel("Score")
    plt.title("Évolution du scores toutes les "+str(n_learn)+"parties d'apprentissage\n(gamma ="+str(gamma)+", alpha="+str(alpha)+", espilon_value="+str(epsilon_value)+")")
    plt.savefig(path_to_save+"simu_greedy_"+str(tot)+"_"+str(gamma)+"_"+str(alpha)+"_"+str(epsilon_value)+".png")
    plt.close()
    
def simu_alpha(n_episodes, gamma, alpha_value, epsilon_value):
    """Simulation et retourne les courbes d'apprentissage"""
    game = SpaceInvaders(display=True)
    eps_profile = EpsilonProfile(epsilon_value, epsilon_value)
    
    for i in range (len(alpha_value)):
        controller = QAgent(game, eps_profile, gamma, alpha_value[i][0])
        scores = controller.learn(game, n_episodes)
        les_x=[i+1 for i in range(n_episodes)]
        plt.plot(les_x, scores, 'xb-', color=alpha_value[i][1])
        
    plt.xlabel("Nb de parties")
    plt.ylabel("Score")    
    plt.title("Influence de alpha sur l'évolution des scores "+str(n_episodes)+"\n(gamma ="+str(gamma)+", espilon_value="+str(epsilon_value)+")")    
    plt.savefig(path_to_save+"simu_alpha_"+str(n_episodes)+"_"+str(gamma)+"_"+str(epsilon_value)+".png")
    plt.close()


def main():
    
    simu_simple(200, 0.8, 0.1, 0.5)
    simu_simple_with_pos_state(200, 0.8, 0.1, 0.5)
    simu_moyenne(200, 0.8, 0.1, 0.5)
    simu_greedy(40, 10, 4, 0.8, 0.1, 0.5)
    alpha_value=[(0.01, 'b'),(0.05,'g'),(0.1,'y'),(0.3, 'violet'), (0.5, 'r')]
    simu_alpha(50, 0.8, alpha_value, 0.5)
    
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
