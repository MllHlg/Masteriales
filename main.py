from ShapeBase import *
import gymnasium as gym
import random

if __name__ == '__main__':

    # Quatre formes disponibles pour l'instant
    env = ShapeEnv(Shape(1))
    episodes = 100
    epsilon = 0.1
    alpha = 0.5
    gamma = 0.2
    for e in range(episodes) :
        if e > 0 : env.reset()
        action = env.get_random_action(epsilon)
        while not env.terminated:
            state = tuple(sorted(env.state))
            point = point_coordinate(action[0])
            reward = env.step(action)
            next_state = tuple(sorted(env.state))
            next_action = env.get_random_action(epsilon)
            if not env.terminated:
                env.shape.Q[state][(point, action[2])] += alpha * (reward + gamma * env.shape.Q[next_state][(point_coordinate(next_action[0]), next_action[2])] - env.shape.Q[state][(point, action[2])])
                action = next_action
            else :
                env.shape.Q[state][(point, action[2])] += alpha * (reward - env.shape.Q[state][(point, action[2])])
        env.render()
    env.afficheEtat()


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
