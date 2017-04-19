import os
import json
import time
from math import *
from datetime import datetime

# import helper functions
from map_server import read_tsv, read_json, haversine

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
#   * path that maximizes speed limit
#   * ...
#   * note: alternative routes should all be independent

map_matched_route_type = "clean_route_latlog"
dijkstras_route_type = "dijkstra_route_latlog"
speed_limit_route_type = "speed_route_latlog"
alt_route_types = [map_matched_route_type,
                   dijkstras_route_type,
                   speed_limit_route_type]

def read_all_alternative_routes(route_types = alt_route_types, fmt = "geojson"):
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
       input: route = list of segments, which are lists of points in route.
       output: direct distance, in meters
    """
    starting_pt = route[0][0]
    destination_pt = route[-1][-1]
    return haversine(*starting_pt, *destination_pt)

def calculate_travel_distance(route):
    """calculate total distance traveled over course of route.
       input: route = list of segments, which are lists of points in route.
       output: total distance, in meters
    """
    return sum([haversine(*pt1, *pt2)
                for route_seg in route
                for pt1, pt2 in zip(route_seg, route_seg[1:])])

def bad_length(route, time):
    """assess whether the trip is too long or short.
       input: route = list of segments, which are lists of points in route.
              time = total amount of time that the trip took, in seconds
       output: boolean
    """
    # determine if travel time is bad
    if time < 60 or time > 90 * 60:
        return True
    
    # determine total number of recorded points
    num_gps_pts = len([pt for route_segment in route
                       for pt in route_segment])
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
       input: route = list of segments, which are lists of points in route.
              djikstra_route = list of segments of points in the optimal route.
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

def filter_alt_routes(alt_routes_by_trip, trip_times):
    # assumes that the first route type is the map matched route
    # and that the second route type is the djikstra's shortest route
    return [alt_routes
            for alt_routes, time in zip(alt_routes_by_trip, trip_times)
            if not bad_route(alt_routes[0], alt_routes[1], time)]

# factors in value function
#   * length of path
#   * aadt / 1000, averaged over segments in path
#   * max speed limit on a segment (SP_LIM)
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

        # make sure that coordinates are in triply nested format
        if feature["geometry"]["type"] == "LineString":
            coords = [coords]

        # remove z component
        coords = map_nested(lambda x: x[:2], coords)
        street_coords_and_props.append((coords, aadt, sp_lim))
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
        trail_data.append(coords)
    return trail_data

def pt_is_between(a, b, c, tol = 10):
    """check if GPS point c is in between a and b, using triangle inequality.
       input: a, b, c = GPS points
              tol = max allowable diff, in meters
    """
    return abs(haversine(*a, *c) + haversine(*c, *b) - haversine(*a, *b)) < tol

def closest_street_id_to_pt(pt, street_coords_and_props):
    for i, (street_coords, _, _) in enumerate(street_coords_and_props):
        for segment in street_coords:
            for pt1_, pt2_ in zip(segment, segment[1:]):
                if pt_is_between(pt1_, pt2_, pt):
                    return i
    return None

def closest_trail_id_to_pt(pt, bike_trails):
    for i, (trail_coords) in enumerate(bike_trails):
        for segment in trail_coords:
            for pt1_, pt2_ in zip(segment, segment[1:]):
                if pt_is_between(pt1_, pt2_, pt):
                    return i
    return None

def proportion_route_on_trail(route, bike_trails):
    dist_on_trail = sum([haversine(*pt1, *pt2)
                         for segment in route
                         for pt1, pt2 in zip(segment, segment[1:])
                         if closest_trail_id_to_pt(pt1, bike_trails) != None
                         and closest_trail_id_to_pt(pt2, bike_trails) != None])
    total_dist = calculate_travel_distance(route)
    return dist_on_trail / total_dist

def turns_per_distance(route):
    """calculate the total number of turns per kilometer traveled"""
    total_dist = calculate_travel_distance(route)
    angles_since_last_turn = []
    number_of_turns = 0
    for segment in route:
        for pt1, pt2 in zip(segment, segment[1:]):
            angle = atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
            if any([abs(abs(angle - angle_) - pi / 2) % (2 * pi) < pi * 0.05
                    for angle_ in angles_since_last_turn]):
                angles_since_last_turn = []
                number_of_turns += 1
            angles_since_last_turn.append(angle)
    return number_of_turns / (total_dist / 1000)

def model_max_likelihood(alt_routes_by_trip):
    # determine attributes of each alternative route
    alt_lengths_by_trip = [[calculate_travel_distance(route)
                            for route in alt_routes]
                           for alt_routes in alt_routes_by_trip]

    street_coords_and_props = read_street_data()
    alt_aadts_by_trip = []
    alt_splim_by_trip = []
    for alt_routes in alt_routes_by_trip:
        alt_aadts = []
        alt_splim = []
        for route in alt_routes:
            street_ids = [closest_street_id_to_pt(pt, street_coords_and_props)
                          for segment in route
                          for pt in segment]
            street_ids = list(set(street_ids).difference([None]))
            
            all_aadts = [street_coords_and_props[i][1] for i in street_ids]
            avg_aadt = sum(all_aadts) / len(all_aadts) / 1000
            alt_aadts.append(avg_aadt)

            all_splim = [street_coords_and_props[i][2] for i in street_ids]
            max_splim = max(all_splim)
            alt_splim.append(max_splim)
        alt_aadts_by_trip.append(alt_aadts)
        alt_splim_by_trip.append(alt_splim)

    bike_trails = read_bike_trail_data()
    alt_trail_perc_by_trip = [[proportion_route_on_trail(route, bike_trails)
                               for route in alt_routes]
                              for alt_routes in alt_routes_by_trip]

    alt_turns_by_trip = [[turns_per_distance(route)
                          for route in alt_routes]
                         for alt_routes in alt_routes_by_trip]
    
    params = [0] * 5
    return alt_lengths_by_trip, alt_aadts_by_trip, alt_splim_by_trip, \
           alt_trail_perc_by_trip, alt_turns_by_trip

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
    params = model_max_likelihood(alt_routes_by_trip)
