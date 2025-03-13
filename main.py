from ShapeBase import *
import gymnasium as gym
import random
import matplotlib.pyplot as plt
import time
import os

def run(epsilon, alpha, gamma, episodes: int = 100) -> None:
    env = ShapeEnv(Shape(1))

    rewards_per_episodes = np.zeros(episodes)

    for ep in range(episodes):
        print(f"Épisode : {ep + 1} / {episodes}") # Info sur l'avancé des épisodes
        
        if ep > 0 : env.reset() # Reset de l'environnement

        action = env.get_random_action(epsilon)

        rewards = 0

        while not env.terminated:
            state = tuple(sorted(env.state))
            point = point_coordinate(action[0])
            reward = env.step(action)
            next_state = tuple(sorted(env.state))
            next_action = env.get_random_action(epsilon)

            if not env.terminated:
                env.shape.Q[state][(point, action[2])] += alpha * (reward + gamma * env.shape.Q[next_state][(point_coordinate(next_action[0]), next_action[2])] - env.shape.Q[state][(point, action[2])])
                action = next_action
            else:
                env.shape.Q[state][(point, action[2])] += alpha * (reward - env.shape.Q[state][(point, action[2])])
            
            rewards += reward
        
        rewards_per_episodes[ep] = rewards

        env.render()
    env.afficheEtat()

    # Sauvegarde des courbes
    time_str = time.strftime("%Y%m%d-%H%M%S")
    
    if not os.path.exists("./results"):
        os.makedirs('./results')
    os.makedirs(f'./results/{time_str}')

    # Calcul de la moyenne des rewards obtenus
    mean_rewards = np.zeros(episodes)
    for i in range(episodes):
        mean_rewards[i] = np.mean(rewards_per_episodes[max(0, i-100):(i+1)])

    #plt.plot(rewards_per_episodes, 'b-') # Courbe Bleu : Reward Brut
    plt.plot(mean_rewards, 'r-') # Courbe Rouge : Moyenne des rewards
    plt.savefig(f'./results/{time_str}/rpe.png')
    plt.close()

    plt.plot(env.shape.nb_visits, 'b+') # Affiche le tableau des visites
    plt.savefig(f'./results/{time_str}/nb_visite.png')
    plt.close()

    plt.plot(env.shape.array_rewards, 'b+') # Affiche le tableau des visites
    plt.savefig(f'./results/{time_str}/array_reward.png')
    plt.close()

    return np.mean(mean_rewards)


if __name__ == '__main__':
    
    run(0.1,0.5,0.55, 1000)
    
    ## Cut from point 13, face 2 in direction 3 (east)
    #cut(13, 2, 3)
    #print_infos()
    ## Cut from point 3, face 1 in direction 2 (south)
    #cut(3, 1, 2)
    #print_infos()

    ## final meshing
    remesh()
    gmsh.write("mesh_gmsh.vtk")
    finalize()
