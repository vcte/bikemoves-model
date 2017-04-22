from math import *
import json

# helper functions
def read_tsv(file_name, header = False):
    data = []
    with open(file_name, mode = "r") as f:
        if header:
            # skip the header
            f.readline()

        # parse rest of tab separated file
        for line in f.readlines():
            data.append(line.strip("\r\n").split("\t"))
    return data

def read_json(fname):
    with open(fname, mode = "r") as f:
        return json.load(f)
    
# calculate great circle distance (in meters) between 2 lat, long points
def haversine(lat1, lng1, lat2, lng2):
    # convert to radians
    lat1, lng1, lat2, lng2 = map(radians, map(float, (lat1, lng1, lat2, lng2)))

    # apply formula, r is radius of earth in meters
    r = 6371000
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * \
        sin((lng2 - lng1) / 2) ** 2
    return 2 * r * asin(sqrt(a))

# determine nesting depth of a list, where nest_depth([]) == 1
def nest_depth(obj):
    if type(obj) is not list:
        return 0
    elif len(obj) < 1:
        return 1
    else:
        return 1 + nest_depth(obj[0])
