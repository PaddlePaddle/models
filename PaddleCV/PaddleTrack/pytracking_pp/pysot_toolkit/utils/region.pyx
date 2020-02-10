"""
    @author fangyi.zhang@vipl.ict.ac.cn
"""
# distutils: sources = src/region.c
# distutils: include_dirs = src/

from libc.stdlib cimport malloc, free
from libc.stdio cimport sprintf
from libc.string cimport strlen

cimport c_region

cpdef enum RegionType:
    EMTPY
    SPECIAL
    RECTANGEL
    POLYGON
    MASK

cdef class RegionBounds:
    cdef c_region.region_bounds* _c_region_bounds

    def __cinit__(self):
        self._c_region_bounds = <c_region.region_bounds*>malloc(
                sizeof(c_region.region_bounds))
        if not self._c_region_bounds:
            self._c_region_bounds = NULL
            raise MemoryError()

    def __init__(self, top, bottom, left, right):
        self.set(top, bottom, left, right)

    def __dealloc__(self):
        if self._c_region_bounds is not NULL:
            free(self._c_region_bounds)
            self._c_region_bounds = NULL

    def __str__(self):
        return "top: {:.3f} bottom: {:.3f} left: {:.3f} reight: {:.3f}".format(
                self._c_region_bounds.top,
                self._c_region_bounds.bottom,
                self._c_region_bounds.left,
                self._c_region_bounds.right)

    def get(self):
        return (self._c_region_bounds.top,
                self._c_region_bounds.bottom,
                self._c_region_bounds.left,
                self._c_region_bounds.right)

    def set(self, top, bottom, left, right):
        self._c_region_bounds.top = top
        self._c_region_bounds.bottom = bottom
        self._c_region_bounds.left = left
        self._c_region_bounds.right = right

cdef class Rectangle:
    cdef c_region.region_rectangle* _c_region_rectangle

    def __cinit__(self):
        self._c_region_rectangle = <c_region.region_rectangle*>malloc(
                sizeof(c_region.region_rectangle))
        if not self._c_region_rectangle:
            self._c_region_rectangle = NULL
            raise MemoryError()

    def __init__(self, x, y, width, height):
        self.set(x, y, width, height)

    def __dealloc__(self):
        if self._c_region_rectangle is not NULL:
            free(self._c_region_rectangle)
            self._c_region_rectangle = NULL

    def __str__(self):
        return "x: {:.3f} y: {:.3f} width: {:.3f} height: {:.3f}".format(
                self._c_region_rectangle.x,
                self._c_region_rectangle.y,
                self._c_region_rectangle.width,
                self._c_region_rectangle.height)

    def set(self, x, y, width, height):
        self._c_region_rectangle.x = x
        self._c_region_rectangle.y = y
        self._c_region_rectangle.width = width
        self._c_region_rectangle.height = height

    def get(self):
        """
        return:
            (x, y, width, height)
        """
        return (self._c_region_rectangle.x,
                self._c_region_rectangle.y,
                self._c_region_rectangle.width,
                self._c_region_rectangle.height)

cdef class Polygon:
    cdef c_region.region_polygon* _c_region_polygon

    def __cinit__(self, points):
        """
        args:
            points: tuple of point
            points = ((1, 1), (10, 10))
        """
        num = len(points) // 2
        self._c_region_polygon = <c_region.region_polygon*>malloc(
                sizeof(c_region.region_polygon))
        if not self._c_region_polygon:
            self._c_region_polygon = NULL
            raise MemoryError()
        self._c_region_polygon.count = num
        self._c_region_polygon.x = <float*>malloc(sizeof(float) * num)
        if not self._c_region_polygon.x:
            raise MemoryError()
        self._c_region_polygon.y = <float*>malloc(sizeof(float) * num)
        if not self._c_region_polygon.y:
            raise MemoryError()

        for i in range(num):
            self._c_region_polygon.x[i] = points[i*2]
            self._c_region_polygon.y[i] = points[i*2+1]

    def __dealloc__(self):
        if self._c_region_polygon is not NULL:
            if self._c_region_polygon.x is not NULL:
                free(self._c_region_polygon.x)
                self._c_region_polygon.x = NULL
            if self._c_region_polygon.y is not NULL:
                free(self._c_region_polygon.y)
                self._c_region_polygon.y = NULL
            free(self._c_region_polygon)
            self._c_region_polygon = NULL

    def __str__(self):
        ret = ""
        for i in range(self._c_region_polygon.count-1):
            ret += "({:.3f} {:.3f}) ".format(self._c_region_polygon.x[i],
                    self._c_region_polygon.y[i])
        ret += "({:.3f} {:.3f})".format(self._c_region_polygon.x[i],
                self._c_region_polygon.y[i])
        return ret

def vot_overlap(polygon1, polygon2, bounds=None):
    """ computing overlap between two polygon
    Args:
        polygon1: polygon tuple of points
        polygon2: polygon tuple of points
        bounds: tuple of (left, top, right, bottom) or tuple of (width height)
    Return:
        overlap: overlap between two polygons
    """
    if len(polygon1) == 1 or len(polygon2) == 1:
        return float("nan")

    if len(polygon1) == 4:
        polygon1_ = Polygon([polygon1[0], polygon1[1],
                             polygon1[0]+polygon1[2], polygon1[1],
                             polygon1[0]+polygon1[2], polygon1[1]+polygon1[3],
                             polygon1[0], polygon1[1]+polygon1[3]])
    else:
        polygon1_ = Polygon(polygon1)

    if len(polygon2) == 4:
        polygon2_ = Polygon([polygon2[0], polygon2[1],
                             polygon2[0]+polygon2[2], polygon2[1],
                             polygon2[0]+polygon2[2], polygon2[1]+polygon2[3],
                             polygon2[0], polygon2[1]+polygon2[3]])
    else:
        polygon2_ = Polygon(polygon2)

    if bounds is not None and len(bounds) == 4:
        pno_bounds = RegionBounds(bounds[0], bounds[1], bounds[2], bounds[3])
    elif bounds is not None and len(bounds) == 2:
        pno_bounds = RegionBounds(0, bounds[1], 0, bounds[0])
    else:
        pno_bounds = RegionBounds(-float("inf"), float("inf"),
                                  -float("inf"), float("inf"))
    cdef float only1 = 0
    cdef float only2 = 0
    cdef c_region.region_polygon* c_polygon1 = polygon1_._c_region_polygon
    cdef c_region.region_polygon* c_polygon2 = polygon2_._c_region_polygon
    cdef c_region.region_bounds no_bounds = pno_bounds._c_region_bounds[0] # deference
    return c_region.compute_polygon_overlap(c_polygon1,
                                            c_polygon2,
                                            &only1,
                                            &only2,
                                            no_bounds)

def vot_overlap_traj(polygons1, polygons2, bounds=None):
    """ computing overlap between two trajectory
    Args:
        polygons1: list of polygon
        polygons2: list of polygon
        bounds: tuple of (left, top, right, bottom) or tuple of (width height)
    Return:
        overlaps: overlaps between all pair of polygons
    """
    assert len(polygons1) == len(polygons2)
    overlaps = []
    for i in range(len(polygons1)):
        overlap = vot_overlap(polygons1[i], polygons2[i], bounds=bounds)
        overlaps.append(overlap)
    return overlaps


def vot_float2str(template, float value):
    """
    Args:
        tempate: like "%.3f" in C syntax
        value: float value
    """
    cdef bytes ptemplate = template.encode()
    cdef const char* ctemplate = ptemplate
    cdef char* output = <char*>malloc(sizeof(char) * 100)
    if not output:
        raise MemoryError()
    sprintf(output, ctemplate, value)
    try:
        ret = output[:strlen(output)].decode()
    finally:
        free(output)
    return ret
