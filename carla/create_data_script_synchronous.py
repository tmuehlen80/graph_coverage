import pandas as pd
import os
os.chdir('carla')
from src.generate_traffic_data import clean_carla, spawn_scene, run_scene
import carla
os.getcwd()
from datetime import datetime
import time
import random
from tqdm import tqdm
import networkx as nx
import numpy as np
from shapely.geometry import Polygon
os.chdir('..')
os.getcwd()
from graph_creator.MapGraph import MapGraph

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

def to_2d(location):
    return (location.x, location.y)

#clean_carla(world)

client.load_world(random.choice(client.get_available_maps()))

def get_t_coordinate(actor, world_map):
    # Thanks Gemini
    location = actor.get_location()
    # Get the closest waypoint on a driving lane
    waypoint = world_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    # Center of the lane
    lane_center = waypoint.transform.location
    # Right vector of the lane (perpendicular to the forward direction)
    right_vector = waypoint.transform.get_right_vector()
    # Vector from lane center to actor
    offset_vector = location - lane_center
    # Compute t-coordinate: lateral distance (positive to the right of lane center)
    t = right_vector.x * offset_vector.x + right_vector.y * offset_vector.y + right_vector.z * offset_vector.z
    return t


j = 1

# create the lane map graph and store it to file:
world_map = world.get_map()
map_g = MapGraph()
map_g = map_g.create_from_carla_map(world_map)

map_g.store_graph_to_file(f"carla/data/scene_{j}_{str(datetime.now().date())}_map_graph.pickle")

# Get the world and set synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True  # Enable sync mode
settings.fixed_delta_seconds = 0.05  # Set fixed time step
world.apply_settings(settings)

# Get the traffic manager
tm = client.get_trafficmanager(8000)  # Port 8000
tm.set_synchronous_mode(True)  # Make TM sync with simulation


# Get blueprint library
blueprint_library = world.get_blueprint_library()

# Spawn vehicles
spawn_points = world.get_map().get_spawn_points()
random.shuffle(spawn_points)

for i in range(25):
    #vehicle_bp = blueprint_library.filter("vehicle.*")[0]
    spawn_point = spawn_points[i]
    vehicle = world.spawn_actor(random.choice(blueprint_library.filter("vehicle.*")), spawn_point)    
    # Enable autopilot
    vehicle.set_autopilot(True, tm.get_port())  # TM handles driving
    #vehicles.append(vehicle)



#all_waypoints = world_map.generate_waypoints(.1)  # Generate waypoints with 1m spacing
#len(all_waypoints)
#lane_waypoints = [wp for wp in all_waypoints if wp.road_id == road_id and wp.lane_id == lane_id]

tracks = []
n_steps = 500

for i in tqdm(range(n_steps)):
    world.tick()
    for act in world.get_actors().filter("vehicle.*"):
        #act = world.get_actors().filter("vehicle.*")[0]
        row = {}
        row['actor_id'] = act.id
        row['actor_type'] = act.type_id
        row['actor_speed_xyz'] = [act.get_velocity().x, act.get_velocity().y, act.get_velocity().z]
        row['actor_acceleration_xyz'] = [act.get_acceleration().x, act.get_acceleration().y, act.get_acceleration().z]
        row['actor_location_xyz'] = [act.get_location().x, act.get_location().y, act.get_location().z]
        bbox = act.bounding_box.get_local_vertices()
        row['actor_bbox'] = [[corner.x, corner.y, corner.z] for corner in bbox]
        till_lane_end_wps = world_map.get_waypoint(act.get_location()).next_until_lane_end(0.25)
        till_lane_end_wps = [world_map.get_waypoint(act.get_location())] + till_lane_end_wps
        from_lane_start_wps = world_map.get_waypoint(act.get_location()).previous_until_lane_start(0.25)
        from_lane_start_wps.append(world_map.get_waypoint(act.get_location()))
        length_from_lane_start = sum([from_lane_start_wps[i].transform.location.distance(from_lane_start_wps[i + 1].transform.location) for i in range(len(from_lane_start_wps) - 1)])
        length_till_lane_end = sum([till_lane_end_wps[i].transform.location.distance(till_lane_end_wps[i + 1].transform.location) for i in range(len(till_lane_end_wps) - 1)])
        row['distance_till_lane_end'] = length_till_lane_end
        row['distance_from_lane_start'] = length_from_lane_start
        row['t_per_lane_id'] = get_t_coordinate(act, world_map)
        row['lane_id'] = world_map.get_waypoint(act.get_location()).lane_id
        row['road_id'] = world_map.get_waypoint(act.get_location()).road_id
        row['carla_road_s'] = world_map.get_waypoint(act.get_location()).s
        row['light_state'] = act.get_light_state()
        row['timestamp'] = world.get_snapshot().timestamp.elapsed_seconds
        forward_vector = act.get_transform().get_forward_vector()
        row['actor_heading_xyz'] = [act.get_transform().get_forward_vector().x, act.get_transform().get_forward_vector().y, act.get_transform().get_forward_vector().z]
        # Compute speed (dot product of velocity and forward vector)
        speed_lon = act.get_velocity().x * forward_vector.x + act.get_velocity().y * forward_vector.y + act.get_velocity().z * forward_vector.z
        # Compute acceleration (dot product of acceleration and forward vector)
        acceleration_lon = act.get_acceleration().x * forward_vector.x + act.get_acceleration().y * forward_vector.y + act.get_acceleration().z * forward_vector.z
        # add lateral speed and accel data
        row['actor_speed_lon'] = speed_lon
        row['actor_acceleration_lon'] = acceleration_lon
        lat_vector = act.get_transform().get_right_vector()
        speed_lat = act.get_velocity().x * lat_vector.x + act.get_velocity().y * lat_vector.y + act.get_velocity().z * lat_vector.z
        # Compute acceleration (dot product of acceleration and forward vector)
        acceleration_lat = act.get_acceleration().x * lat_vector.x + act.get_acceleration().y * lat_vector.y + act.get_acceleration().z * lat_vector.z
        row['actor_speed_lat'] = speed_lat
        row['actor_acceleration_lat'] = acceleration_lat
        row['lane_width'] = world_map.get_waypoint(act.get_location()).lane_width
        row['lane_type'] = world_map.get_waypoint(act.get_location()).lane_type
    tracks.append(row)
    time.sleep(0.2)



tracks_df = pd.DataFrame(tracks)
tracks_df['map'] = world.get_map().name

tracks_df['scene_id'] = j

tracks_df.to_parquet(f"carla/data/scene_{j}_{str(datetime.now().date())}_tracks.parquet")




# get_t_coordinate(act, world_map)

# #world_map.get_waypoint(act.get_location(), project_to_road=True)

# center_waypoint = world_map.get_waypoint(act.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
# center_waypoint.transform.location.distance(act.get_location())

# center_waypoint.transform.get_right_vector().z
# center_waypoint.s
# act.transform.get_transform()
# act_waypoint = world_map.get_waypoint(act.get_location())

# act_waypoint.s


# # manually check:

# print(center_waypoint.transform.location.x, act_waypoint.transform.location.x)
# print(center_waypoint.transform.location.y, act_waypoint.transform.location.y)
# print(center_waypoint.transform.location.z, act_waypoint.transform.location.z)


# dir(world_map.get_waypoint(act.get_location()))
# dir(world_map.get_waypoint(act.get_location()).left_lane_marking)
# world_map.get_waypoint(act.get_location()).left_lane_marking.width

# world_map.get_waypoint(act.get_location()).left_lane_marking




