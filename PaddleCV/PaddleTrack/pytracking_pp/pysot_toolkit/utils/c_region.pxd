cdef extern from "src/region.h":
    ctypedef enum region_type "RegionType":
        EMTPY
        SPECIAL
        RECTANGEL
        POLYGON
        MASK

    ctypedef struct region_bounds:
        float top
        float bottom
        float left
        float right

    ctypedef struct region_rectangle:
        float x
        float y
        float width
        float height

    # ctypedef struct region_mask:
    #     int x
    #     int y
    #     int width
    #     int height
    #     char *data

    ctypedef struct region_polygon:
        int count
        float *x
        float *y

    ctypedef union region_container_data:
        region_rectangle rectangle
        region_polygon polygon
        # region_mask mask
        int special

    ctypedef struct region_container:
        region_type type
        region_container_data data

    # ctypedef struct region_overlap:
    #     float overlap
    #     float only1
    #     float only2

    # region_overlap region_compute_overlap(const region_container* ra, const region_container* rb, region_bounds bounds)

    float compute_polygon_overlap(const region_polygon* p1, const region_polygon* p2, float *only1, float *only2, region_bounds bounds)
