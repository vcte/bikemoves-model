import os
import json
import time
import random
from math import *
from datetime import datetime

import numpy as np
from sklearn.neighbors import KDTree

# import helper functions
from util import read_tsv, read_json, haversine

# constants
data_directory = "static/data/"

# helper functions
def map_nested(f, itr):
    """apply a function to every element in a 2D nested array"""
    return [[f(pt)
             for pt in lst]
            for lst in itr]

# alternative routes
#   * map matched path (true path)
#   * shortest path, determined by dijkstra's algorithm
#   * path that minimizes speed limit
#   * ...
#   * note: alternative routes should all be independent

map_matched_route_type = "clean_data"
dijkstras_route_type = "shortest_route"
speed_limit_route_type = "speed_route"
second_shortest_type = "second_shortest_route"
min_intersection_type = "minimum_intersections_route"
alt_route_types = [map_matched_route_type,
                   dijkstras_route_type,
                   speed_limit_route_type,
                   second_shortest_type,
                   min_intersection_type]

def read_all_alternative_routes(route_types = alt_route_types, fmt = "geojson"):
    print("reading all route data")
    
    # determine the length, in time, of each trip
    def timestamp_to_sec(date):
        return time.mktime(time.strptime(date, "%m/%d/%Y %I:%M:%S %p"))
    
    trip_csv = read_tsv(data_directory + "trip.csv", header = True)
    times_by_trip = []
    for row in trip_csv:
        trip_id = int(row[1])
        start_time = timestamp_to_sec(row[4])
        end_time = timestamp_to_sec(row[5])
        times_by_trip.extend([0] * (trip_id - len(times_by_trip) + 1))
        times_by_trip[trip_id] = end_time - start_time
    
    alt_routes_by_type = []
    for route_type in alt_route_types:
        route_data = []
        if fmt == "geojson":
            route_json = read_json(data_directory + route_type + "." + fmt)
            for feat in route_json["features"]:
                trip_id = int(feat["properties"]["TRIP_ID"])
                coords  = feat["geometry"]["coordinates"]

                # ensure that coordinates are in triply nested format
                if feat["geometry"]["type"] == "LineString":
                    coords = [coords]

                # convert to latitute, longitude order
                coords = map_nested(lambda x: (x[1], x[0]), coords)

                route_data.extend([None] * (trip_id - len(route_data) + 1))
                route_data[trip_id] = coords
        else:
            pass

        alt_routes_by_type.append(route_data)
    
    return alt_routes_by_type, times_by_trip

# filter trips that are either: 
#   * too short
#      * travel time < 1 minute or > 90 minutes
#      * travel distance < 0.01 mile or > 10 miles
#      * average velocity < 1.5 mph or > 25 mph
#      * fewer than 10 GPS points
#   * recreational / non utilitarian
#      * distance > 5 x direct distance between origin and destination
#      * distance > 3 x djikstra distance
#      ? 2 pts that are > 5 minutes apart are within 100 meters of each other

def calculate_direct_distance(route):
    """calculate distance between start and end point of route.
       input: route = list of paths, which are lists of points in route.
       output: direct distance, in meters
    """
    starting_pt = route[0][0]
    destination_pt = route[-1][-1]
    return haversine(*starting_pt, *destination_pt)

def calculate_travel_distance(route):
    """calculate total distance traveled over course of route.
       input: route = list of paths, which are lists of points in route.
       output: total distance, in meters
    """
    return sum([haversine(*pt1, *pt2)
                for path in route
                for pt1, pt2 in zip(path, path[1:])])

def bad_length(route, time):
    """assess whether the trip is too long or short.
       input: route = list of paths, which are lists of points in route.
              time = total amount of time that the trip took, in seconds
       output: boolean
    """
    # determine if travel time is bad
    if time < 60 or time > 90 * 60:
        return True
    
    # determine total number of recorded points
    num_gps_pts = len([pt for route_path in route
                       for pt in route_path])
    if num_gps_pts < 10:
        return True

    # calculate distance (in meters) from start to end point
    direct_distance = calculate_direct_distance(route)
    if direct_distance < 16 or direct_distance > 16000:
        return True

    # calculate average velocity (in meters / s)
    travel_distance = calculate_travel_distance(route)
    travel_velocity = travel_distance / time
    if travel_velocity < 0.67 or travel_velocity > 11.18:
        return True

    return False

def non_utilitarian(route, djikstra_route):
    """assess whether the trip does not seem to have a utilitarian purpose.
       input: route = list of paths, which are lists of points in route.
              djikstra_route = list of paths of points in the optimal route.
       output: boolean
    """
    direct_distance = calculate_direct_distance(route)
    travel_distance = calculate_travel_distance(route)
    djikstra_distance = calculate_travel_distance(djikstra_route)

    # determine if trip is too 'roundabout'
    if travel_distance > 5 * direct_distance:
        return True

    # determine if trip deviates too much from optimal path
    if travel_distance > 3 * djikstra_distance:
        return True

    return False

def bad_route(route, djikstra_route, time):
    return bad_length(route, time) or non_utilitarian(route, djikstra_route)

def unique_routes(alt_routes):
    uniq_routes = []
    for route in alt_routes:
        if route not in uniq_routes:
            uniq_routes.append(route)
    return uniq_routes

def filter_alt_routes(alt_routes_by_trip, trip_times):
    print("filtering routes")
    
    # assumes that the first route type is the map matched route
    # and that the second route type is the djikstra's shortest route
    return [unique_routes(alt_routes)
            for alt_routes, time in zip(alt_routes_by_trip, trip_times)
            if not bad_route(alt_routes[0], alt_routes[1], time)]

# factors in value function
#   * length of path
#   * aadt / 1000, averaged over segments in path
#   * min speed limit on a segment (SP_LIM)
#   * proportion of bike facilities on path
#   * number of turns / mile

def read_street_data():
    """pulls relevant info about streets from json file.
       output: list of (street coords, aadt, speed_limit) triples"""
    street_json = read_json(data_directory + "street.geojson")
    street_coords_and_props = []
    for feature in street_json["features"]:
        # extract information
        aadt = int(feature["properties"]["AADT"])
        sp_lim = int(feature["properties"]["SP_LIM"])
        coords = feature["geometry"]["coordinates"]

        # extract all attributes
        attrs = []
        for key in sorted(feature["properties"].keys()):
            try:
                val = int(feature["properties"][key])
            except BaseException:
                val = 0
            attrs.append(val)

        # make sure that coordinates are in triply nested format
        if feature["geometry"]["type"] == "LineString":
            coords = [coords]

        # convert to latitude, longitude format
        coords = map_nested(lambda x: (x[1], x[0]), coords)

        # remove z component
        coords = map_nested(lambda x: x[:2], coords)
        street_coords_and_props.append((coords, attrs))
    return street_coords_and_props

def read_bike_trail_data():
    """pulls relevant info about bike facilities from json file.
       output: list of coords for each bike facility"""
    trail_json = read_json(data_directory + "bike_trail.geojson")
    trail_data = []
    for feature in trail_json["features"]:
        coords = feature["geometry"]["coordinates"]
        if feature["geometry"]["type"] == "LineString":
            coords = [coords]
        coords = map_nested(lambda x: (x[1], x[0]), coords)
        trail_data.append(coords)
    return trail_data

def pt_is_between(a, b, c, tol = 10):
    """check if GPS point c is in between a and b, using triangle inequality.
       input: a, b, c = GPS points
              tol = max allowable diff, in meters
    """
    return abs(haversine(*a, *c) + haversine(*c, *b) - haversine(*a, *b)) < tol

def get_closest_route_finder(route_list):
    """construct a function to quickly find route that covers a line segment"""
    # map every pt to the street_id for which it belongs
    pt_to_street_id = dict([(pt, i)
                            for i, route
                            in enumerate(route_list)
                            for path in route for pt in path])
    pt_list = list(pt_to_street_id.keys())
    
    # construct kd tree to find nearest street of a pt more efficiently
    x = np.array(pt_list)
    kdt = KDTree(x, metric = 'euclidean')

    def pt_in_route(pt, route):
        return any([pt_is_between(*line_segment, pt)
                    for path in route
                    for line_segment in zip(path, path[1:])])
    
    def closest_route_finder(pt1, pt2):
        closest_st_pts = kdt.query([pt1], k = 10, return_distance = False)
        closest_st_ids = [pt_to_street_id[pt_list[pt_ind]]
                          for pt_ind in closest_st_pts[0]]
        closest_st_ids = list(set(closest_st_ids))
        for st_id in closest_st_ids:
            if pt_in_route(pt1, route_list[st_id]) and \
               pt_in_route(pt2, route_list[st_id]):
                return st_id
        return None

    # return a closure to a function that finds street that best matches a point
    return closest_route_finder

def proportion_route_on_trail(route, trail_finder):
    """calculate percentage of the route spent traveling on bike facilities"""
    dist_on_trail = sum([haversine(*pt1, *pt2)
                         for path in route
                         for pt1, pt2 in zip(path, path[1:])
                         if trail_finder(pt1, pt2) != None])
    total_dist = calculate_travel_distance(route)
    return dist_on_trail / total_dist

def turns_per_distance(route):
    """calculate the total number of turns per kilometer traveled"""
    total_dist = calculate_travel_distance(route)
    angles_since_last_turn = []
    number_of_turns = 0
    for path in route:
        for pt1, pt2 in zip(path, path[1:]):
            angle = atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])

            # determine if difference btwn angles is close to 90 degrees
            if any([abs(abs(angle - angle_) - pi / 2) % (2 * pi) < pi * 0.1
                    for angle_ in angles_since_last_turn]):
                angles_since_last_turn = []
                number_of_turns += 1
            angles_since_last_turn.append(angle)
    return number_of_turns / (total_dist / 1000)

def nearly_identical_pts(pt1, pt2, tol = 3):
    """determine if two (lat, lng) pts are nearly the same"""
    return haversine(*pt1, *pt2) < tol

def nearly_identical_segment(line_segment, line_segment_):
    """determine if two line segments are nearly the same"""
    return (nearly_identical_pts(line_segment[0], line_segment_[0]) and \
            nearly_identical_pts(line_segment[1], line_segment_[1])) or \
           (nearly_identical_pts(line_segment[0], line_segment_[1]) and \
            nearly_identical_pts(line_segment[1], line_segment_[0]))

def line_segment_on_route(line_segment, route_):
    """determine if a line segment lies on a route"""
    return any([nearly_identical_segment(line_segment, line_segment_)
                for path_ in route_
                for line_segment_ in zip(path_, path_[1:])])

def path_size_factor(route, alt_routes):
    """calculates path size factor for route"""
    total_dist = calculate_travel_distance(route)
    return sum([haversine(*pt1, *pt2) / \
                sum([line_segment_on_route((pt1, pt2), route_)
                     for route_ in alt_routes])
                for path in route
                for pt1, pt2 in zip(path, path[1:])]) / total_dist

def generate_attributes(alt_routes_by_trip):
    print("generating route attributes")
    
    # determine attributes of each alternative route
    # calculate the travel distance, in kilometers, for each route
    alt_lengths_by_trip = [[calculate_travel_distance(route) / 1000
                            for route in alt_routes]
                           for alt_routes in alt_routes_by_trip]

    # make a function to find the street that covers a given point
    print(" generating aadt / splim attributes")
    street_coords_and_props = read_street_data()
    st_finder = get_closest_route_finder(
        [st[0] for st in street_coords_and_props])

    street_ids_and_dist_by_trip = []
    for alt_routes in alt_routes_by_trip:
        street_ids_and_dists = []
        for route in alt_routes:
            street_ids_and_dist = [(st_finder(pt1, pt2), haversine(*pt1, *pt2))
                                   for path in route
                                   for pt1, pt2 in zip(path, path[1:])]
            
            street_ids_and_dist = [sid for sid in street_ids_and_dist
                                   if sid[0] != None]
            street_ids_and_dists.append(street_ids_and_dist)
        street_ids_and_dist_by_trip.append(street_ids_and_dists)

    alt_street_attrs_by_trip = []
    for i in range(len(street_coords_and_props[0][1])):
        alt_attr_by_trip = []
        for j, alt_routes in enumerate(alt_routes_by_trip):
            alt_attr = []
            for k, route in enumerate(alt_routes):
                street_ids_and_dist = street_ids_and_dist_by_trip[j][k]

                if len(street_ids_and_dist) > 0:
                    attr_and_dist = [(street_coords_and_props[sid][1][i], dist)
                                     for sid, dist in street_ids_and_dist]
                    wgt_avg_attr = sum([a * b for a, b in attr_and_dist]) /\
                                   sum([dist for _, dist in attr_and_dist])
                else:
                    wgt_avg_attr = 0
                alt_attr.append(wgt_avg_attr)
            alt_attr_by_trip.append(alt_attr)
        alt_street_attrs_by_trip.append(alt_attr_by_trip)
    
##    alt_aadts_by_trip = []
##    alt_splim_by_trip = []
##    for alt_routes in alt_routes_by_trip:
##        alt_aadts = []
##        alt_splim = []
##        for route in alt_routes:
            # get the street id associated with every segment in the route
            # paired with the length of the segment
##            street_ids_and_dist = [(st_finder(pt1, pt2), haversine(*pt1, *pt2))
##                                   for path in route
##                                   for pt1, pt2 in zip(path, path[1:])]

            # remove pairs that could not be mapped to a street
##            street_ids_and_dist = [sid for sid in street_ids_and_dist
##                                   if sid[0] != None]

            # calculate the average weighted aadt
##            if len(street_ids_and_dist) > 0:
##                # list of (aadt, length) pairs for each street segment
##                aadt_and_dists = [(street_coords_and_props[sid[0]][1], sid[1])
##                                  for sid in street_ids_and_dist]
##                wgt_avg_aadt = sum([a * b for a, b in aadt_and_dists]) /\
##                               sum([dist for _, dist in aadt_and_dists])
##            else:
##                wgt_avg_aadt = 0
##            alt_aadts.append(wgt_avg_aadt / 1000)
##
##            # calculate max posted speed limit
##            if len(street_ids_and_dist) > 0:
##                all_splim = [street_coords_and_props[sid[0]][2]
##                             for sid in street_ids_and_dist]
##                max_splim = max(all_splim) / 10
##            else:
##                max_splim = 0
##            alt_splim.append(max_splim)
            
##        alt_aadts_by_trip.append(alt_aadts)
##        alt_splim_by_trip.append(alt_splim)

    print(" generating bike facility attributes")
    bike_trails = read_bike_trail_data()
    trail_finder = get_closest_route_finder(bike_trails)
    alt_trail_perc_by_trip = [[proportion_route_on_trail(route, trail_finder)
                               for route in alt_routes]
                              for alt_routes in alt_routes_by_trip]

    alt_turns_by_trip = [[turns_per_distance(route)
                          for route in alt_routes]
                         for alt_routes in alt_routes_by_trip]

    alt_psf_by_trip = [[path_size_factor(route, alt_routes)
                                 for route in alt_routes]
                                for alt_routes in alt_routes_by_trip]

    all_attr_by_trip = list(zip(alt_lengths_by_trip, alt_trail_perc_by_trip,
                                alt_turns_by_trip, *alt_street_attrs_by_trip))
    
    return all_attr_by_trip, alt_psf_by_trip

def save_attrs(all_attr_by_trip, alt_psf_by_trip):
    attr_list = [[list(attrs) for attrs in zip(*alt_attrs + (psfs, ))]
                 for alt_attrs, psfs in zip(all_attr_by_trip, alt_psf_by_trip)]
    attr_file = data_directory + "attrs.json"
    with open(attr_file, mode = "w", encoding = "utf-8") as f:
        json.dump(attr_list, f, indent = 3)

def read_attrs(attr_file = data_directory + "attrs.json"):
    with open(attr_file, mode = "r", encoding = "utf-8") as f:
        attr_list = json.load(f)
    all_attr_by_trip = []; alt_psf_by_trip = []
    for alt_attrs in attr_list:
        attrs = list(zip(*alt_attrs))
        all_attr_by_trip.append(attrs[:-1])
        alt_psf_by_trip.append(attrs[-1])
    return all_attr_by_trip, alt_psf_by_trip

def calculate_means_and_vars(all_attr_by_trip):
    counts = [0] * len(all_attr_by_trip[0])
    means  = [0] * len(all_attr_by_trip[0])
    m2s    = [0] * len(all_attr_by_trip[0])
    for alt_attrs in all_attr_by_trip:
        for i, attrs in enumerate(alt_attrs):
            for x in attrs:
                counts[i] += 1
                delta = x - means[i]
                means[i] += delta / counts[i]
                delta2 = x - means[i]
                m2s[i] += delta * delta2
    return means, [m2 / (n + 0.001) + 0.001 for n, m2 in zip(counts, m2s)]

def model_max_likelihood(all_attr_by_trip, alt_psf_by_trip):
    print("maximizing log likelihood")

    # normalize all attributes
    means, variances = calculate_means_and_vars(all_attr_by_trip)
    print(means, variances)
    for k, alt_attrs in enumerate(all_attr_by_trip):
        for i, attrs in enumerate(alt_attrs):
            normalized_attrs = []
            for x in attrs:
                normalized_attrs.append((x - means[i]) / sqrt(variances[i]))
            all_attr_by_trip[k][i] = tuple(normalized_attrs)
    
    def value(param, attrs):
        return sum([p * w for p, w in zip(param, attrs)])

    def rescale(params):
        dot_prod = sqrt(sum([p * p for p in params]))
        if dot_prod > 0:
            return [p / dot_prod for p in params]
        else:
            return params
    
    def calc_log_likelihood(params):
        total_ll = 0
        derivatives = [0] * len(params)
        for all_alt_attrs, alt_psf in zip(all_attr_by_trip, alt_psf_by_trip):
            exps = [exp(value(params, attrs) + log(psf))
                    for attrs, psf in zip(zip(*all_alt_attrs), alt_psf)]
            total_ll += log(exps[0] / sum(exps))

            dx = [all_alt_attrs[i][0] -
                  sum([alt_attrs[i] * exp for alt_attrs, exp
                       in zip(zip(*all_alt_attrs), exps)]) / sum(exps)
                  for i in range(len(params))]
            dx = rescale(dx)
            derivatives = [derivatives[i] + dx[i] for i in range(len(params))]
            
        return total_ll, derivatives

    eps = 0.00001
    params = [0.1] * len(all_attr_by_trip[0])
    params = rescale(params)
    for _ in range(1000):
        log_likelihood, derivative = calc_log_likelihood(params)
        derivative = rescale(derivative)
        delta = [0.1 * derivative[i] for i in range(len(params))]

        # halt if the update vector is close to 0
        if all([d < eps for d in delta]):
            break

        # otherwise update parameters
        params = [params[i] + delta[i] for i in range(len(delta))]
        params = rescale(params)
    print("final ll: " + str(log_likelihood))
    print("final params: " + str(params))
    return params

if __name__ == "__main__":
    # get all alternative routes, and time taken, for each trip
    alt_routes_by_type, times_by_trip = read_all_alternative_routes()

    # reformat data
    alt_routes_by_trip = list(zip(*alt_routes_by_type))
    alt_routes_by_trip, trip_times = zip(* \
        [(alt_routes, time)
         for (alt_routes, time) in zip(alt_routes_by_trip, times_by_trip)
         if not any([route is None for route in alt_routes])])

    # filter out trips
    alt_routes_by_trip = filter_alt_routes(alt_routes_by_trip, trip_times)

    # find parameters that maximize model
    all_attr_by_trip, alt_psf_by_trip = generate_attributes(alt_routes_by_trip)

##    all_attr_by_trip, alt_psf_by_trip = read_attrs()
    params = model_max_likelihood(all_attr_by_trip, alt_psf_by_trip)

# TODO
#  * remove identical alternatives
#  * add more alternative routes
