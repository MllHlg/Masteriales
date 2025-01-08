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

def get_voisins(point, angles) : 
    query = Query()
    segments = query.adjacent_curves(point)
    voisins = []
    for seg in segments :
        voisins.extend([i for i in query.adjacent_points(seg) if i != point and i in angles])
    return voisins

def get_chemin(point, chemin, angles) :
    query = Query()
    coorDep = point_coordinate(point)
    segments = query.adjacent_curves(point)
    if len(chemin) > 1 :
        for seg in segments :
            angle = [i for i in query.adjacent_points(seg) if i != point and i in angles]
            if len(angle) > 0 :
                if int(angle[0]) not in chemin : 
                    chemin.append(int(angle[0]))
                    get_chemin(angle[0], chemin, angles)
    else :
        other = False
        voisins = get_voisins(point, angles)
        if len(voisins) > 0 :
            for v in voisins:
                coor = point_coordinate(v)
                if ((coor[0] != coorDep[0]) and (coor[0] > coorDep[0])) or other:
                    chemin.append(int(v))
                    get_chemin(v, chemin, angles)
                else : other = True
    return chemin

# Calcul des sommets ayant un angle concave, donc sur lesquels on peu faire une coupe
def init_angles(face_tag) :
    query = Query()
    angles = query.get_corners(face_tag)
    chemin = get_chemin(angles[0], [int(angles[0])], angles)
    angles_chemins = {}
    for i in range(len(angles)) :
        val_Angle = calculate_angle_from_point(point_coordinate(chemin[i]), point_coordinate(chemin[i-1]), point_coordinate(chemin[(i+1)%len(angles)]))
        if val_Angle >= 190. :
            angles_chemins[chemin[i]] = val_Angle
    return angles_chemins

def get_limits_face(face) :
    query = Query()
    min_X = max_X = min_Y = max_Y = None
    for a in query.get_corners(face) :
        coorA = point_coordinate(a)
        if min_X == None :
            min_X = max_X = coorA[0]
            min_Y = max_Y = coorA[1]
        elif min_X > coorA[0] : min_X = coorA[0]
        elif max_X < coorA[0] : max_X = coorA[0]
        elif min_Y > coorA[1] : min_Y = coorA[1]
        elif max_X < coorA[1] : max_X = coorA[1]
    return (max_Y, min_Y, max_X, min_X)


# Renvoi les coupes possibles sous la forme :
# angles de départ
# face concernée
# direction (1 (north), 2 (south), 3 (east), 4 (west))
def get_action_space() :
    query = Query()
    faces = get_face_tags()
    actionSpace = []
    for face in faces :
        limits = get_limits_face(face)
        angles = init_angles(face)
        #print(f"Face : {face}, angles : {angles}")
        for a in angles :
            dir = [1, 2, 3, 4]
            coorA = point_coordinate(a)
            voisins = get_voisins(a, query.get_corners(face))
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
                        if limits[0] == coorA[1] : break
                    case 2 : 
                        if limits[1] == coorA[1] : break
                    case 3 : 
                        if limits[2] == coorA[0] : break
                    case 4 :
                        if limits[3] == coorA[0] : break
                actionSpace.append((int(a), int(face), d))
    #print(f"Action space : {actionSpace}")
    return actionSpace
            
def get_random_action() :
    actions = get_action_space()
    if len(actions) > 0 :
        return actions[random.randint(0, len(actions)-1)]
    else : return False

def get_segment_size(segment) :
    query = Query()
    points = query.adjacent_points(segment)
    coor1 = point_coordinate(points[0])
    coor2 = point_coordinate(points[1])
    if coor1[0] == coor2[0] :
        return abs(coor1[1] - coor2[1])
    else : return abs(coor1[0] - coor2[0])

def get_reward() :
    query = Query()
    reward = 0
    faces = get_face_tags()
    for face in faces :
        points = query.get_corners(face)
        if len(points) != 4 :
            reward -= 100
        else :
            segments = query.adjacent_curves(points[0])
            size1 = get_segment_size(segments[0])
            size2 = get_segment_size(segments[1])
            reward += 100 / (max(size1, size2) / min(size1, size2))
    return reward

def step() :
    i = 1
    while (get_random_action() != False and i<10) :
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