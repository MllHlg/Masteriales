from rlgeom2d import *
import gymnasium as gym
import random

def create_shape_1():
    """ create a first shape"""
    r1 = create_rectangle(0, 0, 10, 10)
    r2 = create_rectangle(5, 5, 10, 3)
    r3 = create_rectangle(-5, 0, 7, 2)
    fuse([r1, r2, r3])

def create_shape_2():
    """ create a first shape"""
    r1 = create_rectangle(0, 0, 10, 5)
    r2 = create_rectangle(0, 0, 5, 10)
    fuse([r1, r2])

def create_shape_3():
    """ create a first shape"""
    r1 = create_rectangle(0, 0, 5, 2)
    r2 = create_rectangle(5, 0, 2, 2.2)
    r3 = create_rectangle(1, 0, 1, 2.4)
    r4 = create_rectangle(2, 0, 2, 3)
    r5 = create_rectangle(5, 0, 2, 2.4)
    fuse([r1, r2, r3, r4, r5])

def create_shape_4():
    """ create a first shape"""
    r1 = create_rectangle(0, 0, 2, 1)
    r2 = create_rectangle(-2, 0, 2, 1.1)
    r3 = create_rectangle(-0.1, -2, 1, 2)
    r4 = create_rectangle(0, 0, 1.1, 2)
    fuse([r1, r2, r3, r4])
# Press the green button in the gutter to run the script.

observation_space = []

# Renvoi les coupes possibles sous la forme :
# angles de départ
# face concernée
# direction (1 (north), 2 (south), 3 (east), 4 (west))
def get_action_space() :
    actionSpace = []
    for face in Forme().get_faces() :
        angles = angles_concaves(face)
        for a in angles :
            dir = [1, 2, 3, 4]
            coorA = point_coordinate(a)
            voisins = get_voisins(a, face.get_points())
            for v in voisins :
                coorV = point_coordinate(v)
                if  coorA[0] > coorV[0] :
                    dir.remove(4)
                elif coorA[0] < coorV[0] :
                    dir.remove(3)
                elif coorA[1] > coorV[1] :
                    dir.remove(2)
                elif coorA[1] < coorV[1] :
                    dir.remove(1)
            for d in dir :
                match d :
                    case 1 : 
                        if face.get_limits()[0] == coorA[1] : break
                    case 2 : 
                        if face.get_limits()[1] == coorA[1] : break
                    case 3 : 
                        if face.get_limits()[2] == coorA[0] : break
                    case 4 :
                        if face.get_limits()[3] == coorA[0] : break
                actionSpace.append((int(a), int(face.get_tag()), d))
    return actionSpace

# Retourne une action aléatoire parmis celles de l'action space            
def get_random_action() :
    actions = get_action_space()
    if len(actions) > 0 :
        return actions[random.randint(0, len(actions)-1)]
    else : return False

# Calcul le reward d'un état
def get_reward() :
    reward = 0
    for face in Forme().get_faces() :
        if not face.isRect():
            reward -= 100
        else :
            limits = face.get_limits()
            size1 = limits[0] - limits[1]
            size2 = limits[2] - limits[3]
            reward += 100 / (max(size1, size2) / min(size1, size2))
    return round(reward, nb_digit_rounding)

def step() :
    i = 1
    while (get_random_action() != False and i<5) :
        action = get_random_action()
        cut(action[0], action[1], action[2])
        print(f"Action n°{i} : {get_reward()}")
        i += 1

def reset() :
    initialize()
    create_shape_1()

def render() :
    return False

if __name__ == '__main__':

    # first, we initialize the gmsh environment
    initialize()
    create_shape_1()
    step()
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
