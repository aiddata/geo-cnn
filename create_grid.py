
import sys
import os
import json
import time

base_dir = os.path.dirname(os.path.realpath(__file__))

tstart = time.time()

# -----------------------------------------------------------------------------


xsize = 0.5
ysize = 0.5

xmin = -180
xmax = 180

ymin = -90
ymax = 90


# -----------------------------------------------------------------------------


xmax = xmax - xsize
ymin = ymin + ysize

ncols = int((xmax - xmin) / xsize) + 1
nrows = int((ymax - ymin) / ysize) + 1

feature_list = []

# start in top left and go row by row
for r in xrange(nrows):
    y = ymax - (r * ysize)

    for c in xrange(ncols):
        x = xmin + (c * xsize)

        cell_id = (r * ncols) + c

        b_xmin = x
        b_xmax = x + xsize
        b_ymin = y - ysize
        b_ymax = y

        # b_bnds = (b_xmin, b_ymin, b_xmax, b_ymax)

        # env = [
        #     [b_xmin, b_ymax],
        #     [b_xmin, b_ymin],
        #     [b_xmax, b_ymin],
        #     [b_xmax, b_ymax]
        # ]

        # geom = {
        #     "type": "Polygon",
        #     "coordinates": [ [
        #         env[0],
        #         env[1],
        #         env[2],
        #         env[3],
        #         env[0]
        #     ] ]
        # }

        geom = {
            "type": "Polygon",
            "coordinates": [ [
                [b_xmin, b_ymax],
                [b_xmin, b_ymin],
                [b_xmax, b_ymin],
                [b_xmax, b_ymax],
                [b_xmin, b_ymax]
            ] ]
        }

        props = {
            "cell_id": cell_id,
            "row": r,
            "column": c,
            "xcenter": b_xmin + (b_xmax - b_xmin) / 2,
            "ycenter": b_ymax - (b_ymax - b_ymin) / 2,
            "xmin": b_xmin,
            "ymin": b_ymin,
            "xmax": b_xmax,
            "ymax": b_ymax
        }

        # add prio id if grid matches prio grid resolution
        if xsize == 0.5 and ysize == 0.5:
            prio_row = nrows - r
            prio_col = c + 1
            props["prio_id"] = ((prio_row - 1) * ncols) + prio_col

        feature = {
            "type": "Feature",
            "properties": props,
            "geometry": geom
        }


        feature_list.append(feature)



print "Run time: {0} seconds".format(round(time.time() - tstart), 2)


geo_out = {
    "type": "FeatureCollection",
    "features": feature_list
}


# -----------------------------------------------------------------------------


geo_path = os.path.join(base_dir, "grid_0.5_degree.geojson")
geo_file = open(geo_path, "w")
json.dump(geo_out, geo_file)
geo_file.close()
