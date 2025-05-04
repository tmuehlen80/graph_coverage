import pandas as pd
from src.generate_traffic_data import clean_carla, spawn_scene, run_scene
import carla
import os

os.getcwd()
from datetime import datetime
import time
import random
from tqdm import tqdm

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
settings = world.get_settings()
settings.synchronous_mode = True  # Enable sync mode
settings.fixed_delta_seconds = 0.05  # Set fixed time step
world.apply_settings(settings)

tm = client.get_trafficmanager(8000)  # Port 8000
tm.set_synchronous_mode(True)  # Make TM sync with simulation
n_steps = 500


#for j in range(10, 15):
j = 0
clean_carla(world)
client.load_world(random.choice(client.get_available_maps()))
# Get blueprint library
blueprint_library = world.get_blueprint_library()
# Spawn vehicles
spawn_points = world.get_map().get_spawn_points()
random.shuffle(spawn_points)
for i in range(20):
    spawn_point = spawn_points[i]
    vehicle = world.spawn_actor(random.choice(blueprint_library.filter("vehicle.*")), spawn_point)
    # Enable autopilot
    vehicle.set_autopilot(True, tm.get_port())

tracks = []
world_map = world.get_map()
for i in tqdm(range(n_steps)):
    world.tick()
    for act in world.get_actors().filter("vehicle.*"):
        row = {}
        row["actor_id"] = act.id
        row["actor_type"] = act.type_id
        row["actor_speed_xyz"] = [
            act.get_velocity().x,
            act.get_velocity().y,
            act.get_velocity().z,
        ]
        row["actor_acceleration_xyz"] = [
            act.get_acceleration().x,
            act.get_acceleration().y,
            act.get_acceleration().z,
        ]
        row["actor_location_xyz"] = [
            act.get_location().x,
            act.get_location().y,
            act.get_location().z,
        ]
        bbox = act.bounding_box.get_local_vertices()
        row["actor_bbox"] = [[corner.x, corner.y, corner.z] for corner in bbox]
        row["lane_id"] = world_map.get_waypoint(act.get_location()).lane_id
        row["road_id"] = world_map.get_waypoint(act.get_location()).road_id
        row["light_state"] = act.get_light_state()
        row["timestamp"] = world.get_snapshot().timestamp.elapsed_seconds
        forward_vector = act.get_transform().get_forward_vector()
        row["actor_heading_xyz"] = [
            act.get_transform().get_forward_vector().x,
            act.get_transform().get_forward_vector().y,
            act.get_transform().get_forward_vector().z,
        ]
        # Compute speed (dot product of velocity and forward vector)
        speed = (
            act.get_velocity().x * forward_vector.x
            + act.get_velocity().y * forward_vector.y
            + act.get_velocity().z * forward_vector.z
        )
        # Compute acceleration (dot product of acceleration and forward vector)
        acceleration = (
            act.get_acceleration().x * forward_vector.x
            + act.get_acceleration().y * forward_vector.y
            + act.get_acceleration().z * forward_vector.z
        )
        row["actor_speed"] = speed
        row["actor_acceleration"] = acceleration
        tracks.append(row)
        time.sleep(0.2)

tracks_df = pd.DataFrame(tracks)
tracks_df["map"] = world.get_map().name
tracks_df["scene_id"] = j
tracks_df.to_parquet(f"data/scene_{j}_{str(datetime.now().date())}.parquet")



tracks_df.head(2).T

tracks_df.timestamp.value_counts()
tracks_df.groupby("actor_id").mean(numeric_only=True)

for _ in range(500):  # Run for 500 ticks
    world.tick()  # Advances the simulation


# The following loop runs quite unstable. Hence should be babysitted and restarted if needed.
n_tracks = 10
for i in range(n_tracks):
    print(f"{i} scene from {n_tracks}:")
    world, tm = spawn_scene(8, client, seed=i, sync_mode=False)
    tracks_df = run_scene(world, 50)
    tracks_df["scene_id"] = i
    tracks_df.to_parquet(f"data/scene_{i}_{str(datetime.now().date())}.parquet")
    clean_carla(world)
