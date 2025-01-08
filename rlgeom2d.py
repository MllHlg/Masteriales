import gmsh
import math

""""global value to indicate how to round floating numbers"""
nb_digit_rounding = 4

def initialize():
    """ Initialize the gmsh module"""
    gmsh.initialize()
    gmsh.clear()
    # to avoid messages on the console
    gmsh.option.setNumber("General.Verbosity", 0)
    # gmsh.option.setNumber("Mesh.Algorithm", 3) # minimal delaunay
    # 11 <- quad frame based

def create_rectangle(x, y, dx, dy):
    """ create a 2D rectangle from left bottom corner (x,y) and
    side size dx in X direction and dy in Y direction.
    dx and dy must be strictly positive"""
    assert dx > 0
    assert dy > 0
    return gmsh.model.occ.add_rectangle(x, y, 0, dx, dy)


def fuse(pieces):
    """ fuses the input given list of pieces to return the
    Boolean union of all of those pieces """
    tool = [(2, pieces[0])]
    objs = []
    for i in range(1, len(pieces)):
        objs.append((2, pieces[i]))
    op = gmsh.model.occ.fuse(objs, tool)
    gmsh.model.occ.synchronize()
    return op[0][0][1]


def get_cell_tags(dim):
    info = gmsh.model.occ.getEntities(dim)
    tags = []
    for i, j in info:
        tags.append(j)
    return tags


def get_point_tags():
    """" returns the tags of all the vertices"""
    return get_cell_tags(0)


def get_edge_tags():
    """" returns the tags of all the edges"""
    return get_cell_tags(1)


def get_face_tags():
    """" returns the tags of all the faces"""
    return get_cell_tags(2)


def get_end_points(curve_tag):
    """ returns the two end points (extremities) of an edge"""
    data = gmsh.model.getAdjacencies(1, curve_tag)
    return data[1]


class Query:
    def __init__(self):
        self.__points = get_point_tags()
        self.__edges = get_edge_tags()
        self.__l2p = {self.__edges[i]: get_end_points(self.__edges[i]) for i in range(len(self.__edges))}

    def adjacent_curves(self, point_tag):
        curves = []
        for adj in self.__l2p:
            if point_tag in self.__l2p[adj]:
                curves.append(adj)
        return curves

    def adjacent_points(self, curve_tag):
        points = []
        for adj in self.__l2p:
            if adj is curve_tag:
                points.append(self.__l2p[adj][0])
                points.append(self.__l2p[adj][1])
        return points

    def get_curves(self, face_tag):
        return gmsh.model.occ.get_curve_loops(face_tag)[1][0]

    def get_corners(self, face_tag):
        curves = gmsh.model.getAdjacencies(2, face_tag)[1]
        all_points = get_point_tags()
        corners = []
        for i_curve in range(0, len(curves)):
            j_index = i_curve - 1
            if i_curve == 0:
                j_index = len(curves) - 1
            prev_pnts = get_end_points(curves[j_index])
            current_pnts = get_end_points(curves[i_curve])
            common_pnt = 0
            if prev_pnts[0] == current_pnts[0]:
                common_pnt = current_pnts[0]
            elif prev_pnts[0] == current_pnts[1]:
                common_pnt = current_pnts[1]
            elif prev_pnts[1] == current_pnts[0]:
                common_pnt = current_pnts[0]
            else:
                common_pnt = current_pnts[1]
            corners.append(common_pnt)

        return corners


def point_coordinate(tag):
    """:return (x,y) the location of the point 'tag'"""
    bb = gmsh.model.occ.getBoundingBox(0, tag)
    return (round(bb[0], nb_digit_rounding),
            round(bb[1], nb_digit_rounding))

def cut(point_tag, face_tag, dir):
    """ Cut the face face_tag starting from point-tag in direction dir
        dir can be 1 (north), 2 (south), 3 (east), 4 (west)  """
    to_infinity = 0
    coord = point_coordinate(point_tag)
    if dir == 1:
        to_infinity = gmsh.model.occ.add_point(coord[0], coord[1] + 100, 0)
    elif dir == 2:
        to_infinity = gmsh.model.occ.add_point(coord[0], coord[1] - 100, 0)
    elif dir == 3:
        to_infinity = gmsh.model.occ.add_point(coord[0] + 100, coord[1], 0)
    else:
        to_infinity = gmsh.model.occ.add_point(coord[0] - 100, coord[1], 0)

    cut_edge = gmsh.model.occ.addLine(point_tag, to_infinity)
    gmsh.model.occ.fragment([(2, face_tag)], [(1, cut_edge)])
    gmsh.model.occ.synchronize()
    for e in get_edge_tags():
        if len(gmsh.model.getAdjacencies(1, e)[0]) == 0:
            gmsh.model.occ.remove([(1, e)], True)
    gmsh.model.occ.synchronize()

def finalize():
    """ Finalize the gmsh call """
    gmsh.fltk.run()
    gmsh.finalize()

def center(face_tag):
    return gmsh.model.occ.getCenterOfMass(2, face_tag)

def area(face_tag):
    return gmsh.model.occ.getMass(2, face_tag)

def remesh():
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.clear([])
    gmsh.model.mesh.generate(2)  # gives the coarse mesh only

def print_infos():
    print("---------------------------------")
    print("Faces ", get_face_tags())
    for f_tag in get_face_tags():
        q = Query()
        print("Face ", f_tag,", center ", center(f_tag),", and area ", area(f_tag))
        print(" --> curves ", q.get_curves(f_tag))
        print(" --> points ", q.get_corners(f_tag))


# Calcul de l'angle d'un sommet grâce aux sommets voisins
def calculate_angle_from_point(p, a, b):
    """
    Calcule l'angle formé au point P entre deux segments AP et BP,
    en tenant compte des angles > 180°.

    :param p: Coordonnées du point P (tuple: (x, y)).
    :param a: Coordonnées du point A (tuple: (x, y)).
    :param b: Coordonnées du point B (tuple: (x, y)).
    :return: Angle en degrés (entre 0° et 360°).
    """
    # Vecteurs
    u = [a[0] - p[0], a[1] - p[1]]  # Vecteur AP
    v = [b[0] - p[0], b[1] - p[1]]  # Vecteur BP

    # Produit scalaire
    dot_product = u[0] * v[0] + u[1] * v[1]

    # Normes des vecteurs
    norm_u = math.sqrt(u[0]**2 + u[1]**2)
    norm_v = math.sqrt(v[0]**2 + v[1]**2)

    # Calcul du cosinus de l'angle
    cos_theta = dot_product / (norm_u * norm_v)

    # Angle en radians (toujours entre 0 et π)
    theta = math.acos(cos_theta)

    # Calcul du produit vectoriel (déterminant 2D)
    cross_product = u[0] * v[1] - u[1] * v[0]

    # Ajuster l'angle pour les valeurs > 180°
    if cross_product < 0:  # Sens horaire
        theta = 2 * math.pi - theta

    # Conversion en degrés
    angle_degrees = math.degrees(theta)
    return angle_degrees

# Calcul des limites en x et y de la forme fournie
def get_limits_face(face_tag):
    points_coor = [point_coordinate(a) for a in Query().get_corners(face_tag)]
    min_X = min(coor[0] for coor in points_coor)
    max_X = max(coor[0] for coor in points_coor)
    min_Y = min(coor[1] for coor in points_coor)
    max_Y = max(coor[1] for coor in points_coor)
    return max_Y, min_Y, max_X, min_X

# Génère des identifiants pour les points par rapport à leurs coordonnées et leur associe un angle
def points_id(points) :
    pointsID = {}
    chemin = get_chemin(points[0], [int(points[0])], points)
    for i in range(len(points)) :
        pointsID[point_coordinate(chemin[i])] = calculate_angle_from_point(point_coordinate(chemin[i]), point_coordinate(chemin[i-1]), point_coordinate(chemin[(i+1)%len(points)]))
    return pointsID

# Renvoie les sommets voisins d'un point donné dans une forme donnée
def get_voisins(point, angles):
    segments = Query().adjacent_curves(point)
    return [i for seg in segments for i in Query().adjacent_points(seg) if i != point and i in angles]

# Donne un chemin pour parcourir une forme de point en point afin de calculer les angles des sommets
def get_chemin(point, chemin, angles):
    coorDep = point_coordinate(point)
    segments = Query().adjacent_curves(point)
    if len(chemin) > 1:
        for seg in segments:
            angle = [i for i in Query().adjacent_points(seg) if i != point and i in angles]
            if angle and int(angle[0]) not in chemin:
                chemin.append(int(angle[0]))
                get_chemin(angle[0], chemin, angles)
    else:
        other = False
        for v in get_voisins(point, angles):
            coor = point_coordinate(v)
            if ((coor[0] != coorDep[0] and coor[0] > coorDep[0]) or other):
                chemin.append(int(v))
                get_chemin(v, chemin, angles)
            else:
                other = True      
    return chemin

# Calcul des sommets ayant un angle concave, donc sur lesquels on peu faire une coupe
def angles_concaves(face) :
    return [face.get_point_tag_by_coor(clé) for clé, valeur in face.get_pointsID().items() if valeur > 190.]
