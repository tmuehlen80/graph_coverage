
import carla
import random
import os
import pkg_resources
from pathlib import Path
import shutil
import time
import numpy as np
from pascal_voc_writer import Writer
import queue
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd



# remove all actors when starting over again.
def clean_carla(world) -> None:
    actors = world.get_actors()

    # Filter out only vehicles, walkers, and sensors
    vehicles = actors.filter("vehicle.*")
    walkers = actors.filter("walker.*")
    sensors = actors.filter("sensor.*")

    # Destroy all actors
    for actor_list in [vehicles, walkers, sensors]:
        for actor in actor_list:
            actor.destroy()

    print("All non-static actors removed!")


def spawn_scene(n_actors:int, client, seed = 1000, sync_mode = True):
    """Randomly pick map, select actors and place actors."""
    random.seed(seed)
    client.load_world(random.choice(client.get_available_maps()))
    world = client.get_world()
    # Set up the simulator in synchronous mode
    if sync_mode:
        settings = world.get_settings()
        # Enables synchronous mode
        settings.synchronous_mode = True 
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
    print(world.get_map().name)
    map = world.get_map()
    spawn_points = map.get_spawn_points()
    vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
    for i in range(n_actors):
        world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))
    for vehicle in world.get_actors().filter('*vehicle*'):
        vehicle.set_autopilot(True)
    return world



def run_scene(world, n_steps):
    tracks = []
    world_map = world.get_map()
    for i in tqdm(range(n_steps)):
        #update_spectator()
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
    tracks_df = pd.DataFrame(tracks)
    tracks_df['map'] = world.get_map().name
    # tracks_df.to_parquet("tracks.parquet")
    return tracks_df
