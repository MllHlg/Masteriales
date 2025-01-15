from ShapeBase import *
import gymnasium as gym
import random

if __name__ == '__main__':

    # Quatre formes disponibles pour l'instant
    env = ShapeEnv(Shape(1))
    episodes = 20
    for e in range(episodes) :
        if e > 0 : env.reset()
        action = env.get_random_action()
        while not env.terminated:
            reward = env.step(action)
            #print("Reward: ", reward)
            action = env.get_random_action()
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
