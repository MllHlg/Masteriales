#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import random
from rlgeom2d import *

class Shape:
    def __init__(self, num_shape):
        self.start = num_shape
        self.start_shape()
        self.__face_tags = get_face_tags()
        self.__faces = [Face(tag) for tag in self.__face_tags]
        self.nb_visits = []
        self.array_rewards = []
        self.array_states = []
        self.Q = {}

    def get_start(self):
        assert self.start is not None
        return self.start
    
    def get_action_space(self) :
        actionSpace = []
        for face in self.__faces :
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

    def get_reward(self) :
        reward = 0
        for face in self.__faces :
            if not face.isRect():
                reward -= 100
            else :
                limits = face.get_limits()
                size1 = limits[0] - limits[1]
                size2 = limits[2] - limits[3]
                reward -= (max(size1, size2) / min(size1, size2))
        return round(reward, nb_digit_rounding)

    def start_shape(self) :
        initialize()
        match self.start :
            case 1 : self.create_shape_1()
            case 2 : self.create_shape_2()
            case 3 : self.create_shape_3()
            case 4 : self.create_shape_4()
        self.update()

    def update(self) :
        self.__face_tags = get_face_tags()
        self.__faces = [Face(tag) for tag in self.__face_tags]

    def create_shape_1(self):
        """ create a first shape"""
        r1 = create_rectangle(0, 0, 10, 10)
        r2 = create_rectangle(5, 5, 10, 3)
        r3 = create_rectangle(-5, 0, 7, 2)
        fuse([r1, r2, r3])

    def create_shape_2(self):
        """ create a first shape"""
        r1 = create_rectangle(0, 0, 10, 5)
        r2 = create_rectangle(0, 0, 5, 10)
        fuse([r1, r2])

    def create_shape_3(self):
        """ create a first shape"""
        r1 = create_rectangle(0, 0, 5, 2)
        r2 = create_rectangle(5, 0, 2, 2.2)
        r3 = create_rectangle(1, 0, 1, 2.4)
        r4 = create_rectangle(2, 0, 2, 3)
        r5 = create_rectangle(5, 0, 2, 2.4)
        fuse([r1, r2, r3, r4, r5])

    def create_shape_4(self):
        """ create a first shape"""
        r1 = create_rectangle(0, 0, 2, 1)
        r2 = create_rectangle(-2, 0, 2, 1.1)
        r3 = create_rectangle(-0.1, -2, 1, 2)
        r4 = create_rectangle(0, 0, 1.1, 2)
        fuse([r1, r2, r3, r4])

    def visit(self,state):
        ind = self.array_states.index(state)
        self.nb_visits[ind] += 1
        self.array_rewards[ind] += self.get_reward()
        return False


class Face :
    def __init__(self, tag):
        self.__tag = tag
        self.__points = Query().get_corners(self.__tag)
        self.__edges = Query().get_curves(self.__tag)
        self.__limits = get_limits_face(self.__tag)
        self.__pointsID = points_id(self.__points)

    def get_tag(self) :
        return self.__tag

    def get_points(self) :
        return self.__points
    
    def get_pointsID(self):
        return self.__pointsID
    
    def get_limits(self) :
        return self.__limits
    
    # Vérifie si la face est un rectangle ou non 
    def isRect(self) :
        Rect = True
        for point in self.__points :
            coord = point_coordinate(point)
            if coord[0] in self.__limits or coord[1] in self.__limits : break
            Rect = False
        if self.__limits[0] == self.__limits[1] or self.__limits[2] == self.__limits[3] : Quadr = False
        return Rect


class ShapeEnv(gym.Env):
    """ Environnement pour forme"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, s):
        super(ShapeEnv, self).__init__()
        self.shape = s
        self.state = []
        # Initialisation de la récompense
        self.reward = 0
        self.steps = 0

        self.terminated = False

    def get_random_action(self) :
        self.action_space = self.shape.get_action_space()
        if len(self.action_space) > 0 :
            self.action_from_Q()
            return self.action_space[random.randint(0, len(self.action_space)-1)]
        else : self.terminated = True
    
    def action_from_Q(self) :
        state = tuple(sorted(self.state))
        if state in self.shape.Q :
            actions = self.shape.Q[state]
            max_action = max(actions, key=actions.get)
            return (get_point_tag_by_coord(max_action[0]), max_action[1])
        else :
            actions = {}
            for point_tag,_,direction in self.action_space :
                actions[(point_coordinate(point_tag), direction)] = 0.
            self.shape.Q[state] = actions
            random_key = random.choice(list(actions.keys()))
            return (get_point_tag_by_coord(random_key[0]), random_key[1])

    def reset(self):
        self.close()
        self.shape.start_shape()
        self.state = []
        self.steps = 0
        self.terminated = False

    def step(self, action):
        coord_point = point_coordinate(action[0])
        cut(action[0], action[1], action[2])
        self.steps += 1
        self.shape.update()
        reward = self.shape.get_reward()
        self.state.append(trier_points((coord_point, get_new_segment(coord_point, action[2]))))
        if sorted(self.state) in self.shape.array_states :
            self.shape.visit(sorted(self.state))
        else :
            self.shape.array_states.append(sorted(self.state).copy())
            self.shape.nb_visits.append(1)
            self.shape.array_rewards.append(reward)
        return reward

    def render(self):
        remesh()

    def afficheEtat(self) :
        #print(self.shape.array_states)
        print(f"Tableau nb visits : {self.shape.nb_visits}")
        self.shape.array_rewards = np.round(np.divide(self.shape.array_rewards, self.shape.nb_visits), decimals=nb_digit_rounding).tolist()
        print(f"Tableau des rewards : {self.shape.array_rewards}")
        print(f"Tableau Q : {self.shape.Q}")

    def close(self):
        gmsh.write("mesh_gmsh.vtk")
        finalize()
