from ShapeBase import *
import gymnasium as gym
import random

if __name__ == '__main__':

    env = ShapeEnv(Shape(1))

    env.reset()
    env.render()
    action = env.get_random_action()
    while not env.terminated:
        reward = env.step(action)
        env.render()
        print("Reward: ", reward)
        action = env.get_random_action()


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
