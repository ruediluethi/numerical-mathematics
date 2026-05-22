import numpy as np
import matplotlib as plt
from matplotlib.patches import Polygon

def mitternacht(a, b, c):
  inside_sqrt = b*b - 4*a*c

  if inside_sqrt <= 0:
    return None, None

  x_1 = (-b + np.sqrt(inside_sqrt))/(2*a)
  x_2 = (-b - np.sqrt(inside_sqrt))/(2*a)

  return x_1, x_2

def line_from_two_points(x_1, y_1, x_2, y_2):
    m = (y_2 - y_1) / (x_2 - x_1)
    b = y_1 - m*x_1
    return m, b # l(x) = mx + b

# l(x) = mx + b
# c(x) = sqrt(r**2 - (x - c_x)**2) + c_y
def circle_line_intersection(c_x, c_y, r, m, b):
    x_1, x_2 = mitternacht(1+m**2, 2*m*(b - c_y) - 2*c_x, -r**2 + c_x**2 + (b - c_y)**2)
    if x_1 is None or x_2 is None:
        return []
    return [x_1, x_2]

def circle_horizontal_intersection(c_x, c_y, r, h, min_bounds=None, max_bounds=None, flip_axis=False):
    if flip_axis:
        swap_c = c_x
        c_x = c_y
        c_y = swap_c

    x_1, x_2 = mitternacht(1, -2*c_x, c_x**2 - r**2 + (h - c_y)**2)
    
    if x_1 is None or x_2 is None:
        return []
    
    if min_bounds is None or max_bounds is None:
        return [x_1, x_2]
    
    intersection_points = []
    if min_bounds < x_1 and x_1 < max_bounds:
        if flip_axis:
            intersection_points.append((h, x_1))
        else:
            intersection_points.append((x_1, h))
    if min_bounds < x_2 and x_2 < max_bounds:
        if flip_axis:
            intersection_points.append((h, x_2))
        else:
            intersection_points.append((x_2, h))
    return intersection_points

# def line_intersection(x_11, y_11, x_12, y_12, x_21, y_21, x_22, y_22, check_bounds=True):
    
    

#     A = np.array([
#         [y_12 - y_11, -(x_12 - x_11)],
#         [y_22 - y_21, -(x_22 - x_21)],
#     ])
#     b = np.array([
#         x_11*(y_12 - y_11) - y_12*(x_12 - x_11),
#         x_21*(y_22 - y_21) - y_22*(x_22 - x_21),
#     ])
#     intersection = np.linalg.solve(A, b).flatten()
#     x = intersection[0]
#     y = intersection[1]

#     # a1, b1 = y_12-y_11, -(x_12-x_11)
#     # a2, b2 = y_22-y_21, -(x_22-x_21)
#     # c1 = a1*x_11 + b1*y_11
#     # c2 = a2*x_21 + b2*y_21
    
#     # D = a1*b2 - a2*b1
#     # if np.isclose(D, 0):
#     #     return None, None  # parallel
    
#     # x = (c1*b2 - c2*b1) / D
#     # y = (a1*c2 - a2*c1) / D
    
#     # if check_bounds:
#     #     in_seg1 = min(x_11,x_12) <= x <= max(x_11,x_12) and min(y_11,y_12) <= y <= max(y_11,y_12)
#     #     in_seg2 = min(x_21,x_22) <= x <= max(x_21,x_22) and min(y_21,y_22) <= y <= max(y_21,y_22)
#     #     if not (in_seg1 and in_seg2):
#     #         return None, None
    
#     return x, y

def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4, check_bounds=True):
    # Ein Vektor ausgehend vom Punkt 1
    # (x - x_1, y - y_1)
    # Zweiter Vektor zwischen Punkt 1 und 2
    # (x_2 - x_1, y_2 - y_1)
    # Beide Vektoren sind parallel zueinander wenn Cross-Product != 0
    # (x - x_1)(y_2 - y_1) - (y - y_1)(x_2 - x_1) != 0
    # => x(y_2 - y_1) - y(x_2 - x_1) = x_1(y_2 - y_1) - y_1(x_2 - x_1)

    # Richtungsvektoren
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3

    denominator = dx1 * dy2 - dy1 * dx2
    if np.isclose(denominator, 0):
        return None, None  # Parallel

    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denominator
    x = x1 + t * dx1
    y = y1 + t * dy1

    if check_bounds:
        in_seg1 = min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)
        in_seg2 = min(x3, x4) <= x <= max(x3, x4) and min(y3, y4) <= y <= max(y3, y4)
        if not (in_seg1 and in_seg2):
            return None, None

    return x, y

def sort_ccw(vertices):
    phi = []
    v = np.array(vertices)
    c = v.mean(axis=0)
    phi = np.arctan2(v[:,1] - c[1], v[:,0] - c[0])
    return v[np.argsort(phi)]
    

class QuadSquare:
    def __init__(self, c_x, c_y, s):
        self.c_x = c_x
        self.c_y = c_y
        self.s = s
        self.intersections = []
        self.children = []
        self.edges = np.zeros((2,2))

    def plot(self, ax, alpha=0.1):
        ax.plot([self.c_x - self.s/2, self.c_x + self.s/2], [self.c_y - self.s/2, self.c_y - self.s/2], 'k-', alpha=alpha)
        ax.plot([self.c_x - self.s/2, self.c_x + self.s/2], [self.c_y + self.s/2, self.c_y + self.s/2], 'k-', alpha=alpha)
        ax.plot([self.c_x - self.s/2, self.c_x - self.s/2], [self.c_y - self.s/2, self.c_y + self.s/2], 'k-', alpha=alpha)
        ax.plot([self.c_x + self.s/2, self.c_x + self.s/2], [self.c_y - self.s/2, self.c_y + self.s/2], 'k-', alpha=alpha)

        if np.sum(self.edges) == 4:
            for child in self.children:
                child.plot(ax)
            return

        # for i in range(2):
        #         x = self.c_x - self.s/2 + self.s*i
        #         for j in range(2):
        #             y = self.c_y + self.s/2 - self.s*j
        #             if self.edges[j,i] == 1:
                        # ax.scatter(x, y, marker='o', facecolors='none', edgecolors=color, linewidths=1, s=self.s**2*500, alpha=0.5)

        
        
        

        if len(self.children) == 0:
            vertices = []
            for i in range(2):
                x = self.c_x - self.s/2 + self.s*i
                for j in range(2):
                    y = self.c_y + self.s/2 - self.s*j
                    if self.edges[j,i] == 1:
                        # ax.plot(x, y, 'b.', alpha=0.1)
                        vertices.append((x, y))

            for k in range(len(self.intersections)):
                vertices.append((self.intersections[k][0], self.intersections[k][1]))

            if len(vertices) > 0:
                vertices = sort_ccw(vertices)
                ax.add_patch(Polygon(vertices, closed=True, fill=False, alpha=1.0))

            for k in range(len(self.intersections)):
                for l in range(k):

                    x_1 = self.intersections[k][0]
                    x_2 = self.intersections[l][0]
                    y_1 = self.intersections[k][1]
                    y_2 = self.intersections[l][1]

                    # ax.plot([x_1, x_2], [y_1, y_2], 'r')

        for child in self.children:
            child.plot(ax)

    def calc_circle_intersection(self, c_x, c_y, r, split_threshold=0.1, ax=None):
        count_inside = 0
        for i in range(2):
            x = self.c_x - self.s/2 + self.s*i
            for j in range(2):
                y = self.c_y + self.s/2 - self.s*j
                d = np.sqrt((x - c_x)**2 + (y - c_y)**2)
                if d < r:
                    self.edges[j,i] = 1
                    count_inside = count_inside + 1


        if len(self.children) > 0:
            for child in self.children:
                child.calc_circle_intersection(c_x, c_y, r, split_threshold=split_threshold, ax=ax)
            return
        
        
        if count_inside == 4:
            return

        crnt_intersections = []
        # upper edge
        crnt_intersections.extend(circle_horizontal_intersection(c_x, c_y, r, self.c_y + self.s/2,
                                                      min_bounds=self.c_x - self.s/2, max_bounds=self.c_x + self.s/2))
        # lower edge
        crnt_intersections.extend(circle_horizontal_intersection(c_x, c_y, r, self.c_y - self.s/2,
                                                      min_bounds=self.c_x - self.s/2, max_bounds=self.c_x + self.s/2))
        # left edge
        crnt_intersections.extend(circle_horizontal_intersection(c_x, c_y, r, self.c_x - self.s/2, flip_axis=True,
                                                      min_bounds=self.c_y - self.s/2, max_bounds=self.c_y + self.s/2))
        # right edge
        crnt_intersections.extend(circle_horizontal_intersection(c_x, c_y, r, self.c_x + self.s/2, flip_axis=True,
                                                      min_bounds=self.c_y - self.s/2, max_bounds=self.c_y + self.s/2))

        # if not ax is None:
        #     for inter in crnt_intersections:
        #         ax.plot(inter[0], inter[1], 'kx')

        # return intersections

        # TODO think about special case if center is inside square
        # if len(self.intersections) == 0:
        #     return

        sagitta_min = None
        sagitta_max = 0
        # every with every is too much, better would be to find the nearest
        for k in range(len(crnt_intersections)):
            for l in range(k):

                x_1 = crnt_intersections[k][0]
                x_2 = crnt_intersections[l][0]
                y_1 = crnt_intersections[k][1]
                y_2 = crnt_intersections[l][1]

                # inter_w = np.abs(x_1 - x_2)
                # inter_h = np.abs(y_1 - y_2)

                x_c = (x_1 + x_2)/2
                y_c = (y_1 + y_2)/2

                # distanz von der mitte der sextante zum kreis
                sagitta = r - np.sqrt((c_x - x_c)**2 + (c_y - y_c)**2)
                # normalized to square size
                sagitta = sagitta / self.s

                if sagitta_min is None or sagitta < sagitta_min:
                    sagitta_min = sagitta
                if sagitta_max is None or sagitta > sagitta_max:
                    sagitta_max = sagitta

                # if not ax is None:
                #     ax.plot([x_1, x_2], [y_1, y_2], 'r:')
                    # m, b = line_from_two_points(x_c, y_c, c_x, c_y)
                    # circle_intersections = circle_line_intersection(c_x, c_y, r, m, b)
                    # for x_i in circle_intersections:
                    #     if self.c_x - self.s/2 < x_i and x_i < self.c_x + self.s/2:
                    #         ax.plot([x_i, x_c], [np.sqrt(r**2 - (x_i - c_x)**2) + c_y, y_c], 'r')

        d = np.sqrt((self.c_x - c_x)**2 + (self.c_y - c_y)**2) # distance from square center to circle center
        # split if approximation is not good enough or circle is completly inside the square
        if sagitta_max > split_threshold or r+d < np.sqrt(2)*self.s:
            self.split(ax=ax)
            for child in self.children:
                child.calc_circle_intersection(c_x, c_y, r, split_threshold=split_threshold, ax=ax)
        else:
            self.intersections.extend(crnt_intersections)

        # if inter_w >= inter_h:
        #     x_1 = intersections[0][0]
        #     x_2 = intersections[1][0]
        #     x_sc = (x_1 + x_2)/2
        #     y_c = np.sqrt(r**2 - (x_c - c_x)**2) + c_y
        #     df = -(x_c - c_x)/np.sqrt(r**2 - (x_c - c_x)**2)
        #     y_1 = df*(x_1 - x_c) + y_c
        #     y_2 = df*(x_2 - x_c) + y_c
        # else:
        #     y_1 = intersections[0][1]
        #     y_2 = intersections[1][1]

        #     y_c = (y_1 + y_2)/2
        #     x_c = np.sqrt(r**2 - (y_c - c_y)**2) + c_x
        #     df = -(y_c - c_y)/np.sqrt(r**2 - (y_c - c_y)**2)

        
        

        # if not ax is None:
        #     if inter_w >= inter_h:
        #         x_ = np.linspace(x_c-self.s, x_c+self.s, 100)
        #         ax.plot(x_, df*(x_ - x_c) + y_c, 'r--') # tangente
        #         ax.plot(x_c, y_c, 'b.') # zentrum
        #         ax.plot(x_1, y_1, 'rx') # schnittpunkt linear
        #         ax.plot(x_2, y_2, 'rx')
        #     else:
        #         y_ = np.linspace(y_c-self.s, y_c+self.s, 100)
        #         ax.plot(df*(y_ - y_c) + x_c, y_, 'r--') # tangente
        #         ax.plot(x_c, y_c, 'b.') # zentrum
        #         # ax.plot(x_1, y_1, 'rx') # schnittpunkt linear
        #         # ax.plot(x_2, y_2, 'rx')
        #         pass
        return self.edges

    def split(self, ax=None):
        # TODO split intersections too, otherwise the children will lose the current intersection
        # for x in np.linspace(self.c_x + self.s/4, self.c_x - self.s/4, 2):
        #     for y in np.linspace(self.c_y + self.s/4, self.c_y - self.s/4, 2):
        for i in range(2):
            x = self.c_x - self.s/4 + self.s/2*i
            for j in range(2):
                y = self.c_y + self.s/4 - self.s/2*j

                s = QuadSquare(x, y, self.s / 2)
                s.edges[j,i] = self.edges[j,i]

                # print(x, y, j,i, self.edges[j,i])

                for k in range(len(self.intersections)):
                    for l in range(k):

                        x_1 = self.intersections[k][0]
                        x_2 = self.intersections[l][0]
                        y_1 = self.intersections[k][1]
                        y_2 = self.intersections[l][1]

                        # upper edge
                        inter_x, inter_y = line_intersection(x_1, y_1, x_2, y_2,
                                                             x - self.s/4, y + self.s/4, x + self.s/4, y + self.s/4)
                        if inter_x is not None and inter_y is not None:
                            s.intersections.append((inter_x, inter_y))
                            if not ax is None:
                                ax.plot(inter_x, inter_y, 'r.')
                        # lower edge
                        inter_x, inter_y = line_intersection(x_1, y_1, x_2, y_2,
                                                             x - self.s/4, y - self.s/4, x + self.s/4, y - self.s/4)
                        if inter_x is not None and inter_y is not None:
                            s.intersections.append((inter_x, inter_y))
                            if not ax is None:
                                ax.plot(inter_x, inter_y, 'r.')
                        # left edge
                        inter_x, inter_y = line_intersection(x_1, y_1, x_2, y_2,
                                                             x - self.s/4, y + self.s/4, x - self.s/4, y - self.s/4)
                        if inter_x is not None and inter_y is not None:
                            s.intersections.append((inter_x, inter_y))
                            if not ax is None:
                                ax.plot(inter_x, inter_y, 'r.')
                        # right edge
                        inter_x, inter_y = line_intersection(x_1, y_1, x_2, y_2,
                                                             x + self.s/4, y + self.s/4, x + self.s/4, y - self.s/4)
                        if inter_x is not None and inter_y is not None:
                            s.intersections.append((inter_x, inter_y))
                            if not ax is None:
                                ax.plot(inter_x, inter_y, 'r.')




                
                self.children.append(s)

        

    @property
    def data(self):
        return self._data
    
    def __repr__(self) -> str:
        return f'QuadSquare(c_x={self.c_x}, c_y={self.c_y}, s={self.s})'