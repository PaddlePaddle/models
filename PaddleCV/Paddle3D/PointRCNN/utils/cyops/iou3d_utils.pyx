#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import cython 
from math import pi, cos, sin
import numpy as np 
cimport numpy as np 


cdef class Point:
    cdef float x, y 
    def __cinit__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        if not isinstance(v, Point):
            return NotImplemented
        return Point(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        if not isinstance(v, Point):
            return NotImplemented
        return Point(self.x - v.x, self.y - v.y)

    def cross(self, v):
        if not isinstance(v, Point):
            return NotImplemented
        return self.x*v.y - self.y*v.x


cdef class Line:
    cdef float a, b, c 
    # ax + by + c = 0
    def __cinit__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a*p.x + self.b*p.y + self.c

    def intersection(self, other):
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a*other.b - self.b*other.a
        return Point(
            (self.b*other.c - self.c*other.b)/w,
            (self.c*other.a - self.a*other.c)/w
        )


@cython.boundscheck(False)
@cython.wraparound(False)
def rectangle_vertices_(x1, y1, x2, y2, r):
    
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    angle = r 
    cr = cos(angle)
    sr = sin(angle)
    # rotate around center
    return (
        Point(
            x=(x1-cx)*cr+(y1-cy)*sr+cx,
            y=-(x1-cx)*sr+(y1-cy)*cr+cy
        ),
        Point(
            x=(x2-cx)*cr+(y1-cy)*sr+cx,
            y=-(x2-cx)*sr+(y1-cy)*cr+cy
        ),
        Point(
            x=(x2-cx)*cr+(y2-cy)*sr+cx,
            y=-(x2-cx)*sr+(y2-cy)*cr+cy
        ),
        Point(
            x=(x1-cx)*cr+(y2-cy)*sr+cx,
            y=-(x1-cx)*sr+(y2-cy)*cr+cy
        )
    )

@cython.boundscheck(False)
@cython.wraparound(False)
def intersection_area(r1, r2):
    # r1 and r2 are in (center, width, height, rotation) representation
    # First convert these into a sequence of vertices

    rect1 = rectangle_vertices_(*r1)
    rect2 = rectangle_vertices_(*r2)

    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = rect1

    # Loop over the edges of the second rectangle
    for p, q in zip(rect2, rect2[1:] + rect2[:1]):
        if len(intersection) <= 2:
            break # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
            intersection, intersection[1:] + intersection[:1],
            line_values, line_values[1:] + line_values[:1]):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0

    return 0.5 * sum(p.x*q.y - p.y*q.x for p, q in zip(intersection, intersection[1:] + intersection[:1]))


def boxes3d_to_bev_(boxes3d):
    """
    Args:
        boxes3d: [N, 7], (x, y, z, h, w, l, ry)
    Return:
        boxes_bev: [N, 5], (x1, y1, x2, y2, ry)
    """
    boxes_bev = np.zeros((boxes3d.shape[0], 5), dtype='float32')
    cu, cv = boxes3d[:, 0], boxes3d[:, 2]
    half_l, half_w = boxes3d[:, 5] / 2, boxes3d[:, 4] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev


def boxes_iou3d(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = boxes3d_to_bev_(boxes_a)
    boxes_b_bev = boxes3d_to_bev_(boxes_b)
    # bev overlap
    num_a = boxes_a_bev.shape[0]
    num_b = boxes_b_bev.shape[0]
    overlaps_bev = np.zeros((num_a, num_b), dtype=np.float32)
    for i in range(num_a):
        for j in range(num_b):
            overlaps_bev[i][j] = intersection_area(boxes_a_bev[i], boxes_b_bev[j])

    # height overlap
    boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, 3]).reshape(-1, 1)
    boxes_a_height_max = boxes_a[:, 1].reshape(-1, 1)
    boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, 3]).reshape(1, -1)
    boxes_b_height_max = boxes_b[:, 1].reshape(1, -1)

    max_of_min = np.maximum(boxes_a_height_min, boxes_b_height_min)
    min_of_max = np.minimum(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = np.clip(min_of_max - max_of_min, a_min=0, a_max=np.inf)
    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).reshape(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).reshape(1, -1)

    iou3d = overlaps_3d / np.clip(vol_a + vol_b - overlaps_3d, a_min=1e-7, a_max=np.inf)
    return iou3d

#if __name__ == '__main__':
#    # (center, width, height, rotation)
#    r1 = (10, 15, 15, 10, 30)
#    r2 = (15, 15, 20, 10, 0)
#    print(intersection_area(r1, r2))
