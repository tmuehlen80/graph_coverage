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

j = 0

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



tracks = []
n_steps = 500

for i in tqdm(range(n_steps)):
    world.tick()
    for act in world.get_actors().filter("vehicle.*"):
        row = {}
        row['actor_id'] = act.id
        row['actor_type'] = act.type_id
        row['actor_speed_xyz'] = [act.get_velocity().x, act.get_velocity().y, act.get_velocity().z]
        row['actor_acceleration_xyz'] = [act.get_acceleration().x, act.get_acceleration().y, act.get_acceleration().z]
        row['actor_location_xyz'] = [act.get_location().x, act.get_location().y, act.get_location().z]
        bbox = act.bounding_box.get_local_vertices()
        row['actor_bbox'] = [[corner.x, corner.y, corner.z] for corner in bbox]
        row['lane_id'] = world_map.get_waypoint(act.get_location()).lane_id
        row['road_id'] = world_map.get_waypoint(act.get_location()).road_id
        row['lane_s'] = world_map.get_waypoint(act.get_location()).s
        row['light_state'] = act.get_light_state()
        row['timestamp'] = world.get_snapshot().timestamp.elapsed_seconds
        forward_vector = act.get_transform().get_forward_vector()
        row['actor_heading_xyz'] = [act.get_transform().get_forward_vector().x, act.get_transform().get_forward_vector().y, act.get_transform().get_forward_vector().z]
        # Compute speed (dot product of velocity and forward vector)
        speed = act.get_velocity().x * forward_vector.x + act.get_velocity().y * forward_vector.y + act.get_velocity().z * forward_vector.z
        # Compute acceleration (dot product of acceleration and forward vector)
        acceleration = act.get_acceleration().x * forward_vector.x + act.get_acceleration().y * forward_vector.y + act.get_acceleration().z * forward_vector.z
        row['actor_speed'] = speed
        row['actor_acceleration'] = acceleration
        tracks.append(row)
        time.sleep(0.2)




tracks_df = pd.DataFrame(tracks)
tracks_df['map'] = world.get_map().name

tracks_df['scene_id'] = j
# tracks_df.describe()

tracks_df.to_parquet(f"carla/data/scene_{j}_{str(datetime.now().date())}_tracks.parquet")



