# Before running this script, make sure to start the CARLA server via nomachine (needs a screen):
# tmuehlen@tmuehlen-HP-Z8-G4-Workstation:~/repos/graph_coverage/CARLA_0.9.15$ ./CarlaUE4.sh 
# use `poetry shell` to activate the environment

import pandas as pd
import os
os.chdir("carla")
from src.generate_traffic_data import clean_carla
import carla
os.getcwd()
from datetime import datetime
import time
import random
from tqdm import tqdm
os.chdir("..")
os.getcwd()
from graph_creator.MapGraph import MapGraph


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


def to_2d(location):
    return (location.x, location.y)


maps = [
    '/Game/Carla/Maps/Town01', 
    '/Game/Carla/Maps/Town01_Opt', 
    '/Game/Carla/Maps/Town02_Opt', 
    '/Game/Carla/Maps/Town02', 
    '/Game/Carla/Maps/Town03', 
    '/Game/Carla/Maps/Town04', # has a lot of offroad actors
    '/Game/Carla/Maps/Town04_Opt', # has a lot of offroad actors
    '/Game/Carla/Maps/Town05_Opt', 
    '/Game/Carla/Maps/Town05', 
    '/Game/Carla/Maps/Town07', 
    # '/Game/Carla/Maps/Town10HD', # problematic. for whatever reason...
    # '/Game/Carla/Maps/Town10HD_Opt', 
    # '/Game/Carla/Maps/Town11/Town11' 
]

n_steps = 300
timeout = 150.0
for map in maps:
    for ijk in range(5):
        client = carla.Client("localhost", 2000)
        client.set_timeout(timeout)
        client.load_world(map)
        world = client.get_world()
        clean_carla(world)
        _ = world.tick()
        print(map)
        # create the lane map graph and store it to file:
        world_map = world.get_map()
        map_g = MapGraph()
        map_g = map_g.create_from_carla_map(world_map)
        dt_specifier = str(datetime.now())
        map_name = map.split("/")[-1]
        scene_name = f"{dt_specifier}_{map_name}"
        map_g.store_graph_to_file(f"carla/data/scene_{scene_name}_map_graph.pickle")
        xodr = world_map.to_opendrive()
        with open(f"carla/data/scene_{scene_name}_map.xodr", "w") as f:
            f.write(xodr)
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
        # Track vehicle behavior states
        vehicle_behaviors = {}
        vehicle_slowdown_timers = {}
        n_vehicles = random.randint(20, 60)
        for i in range(n_vehicles):
            spawn_point = spawn_points[i]
            vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                # Store vehicle reference for behavior control
                vehicle_id = vehicle.id
                # Enable autopilot
                vehicle.set_autopilot(True, tm.get_port())  # TM handles driving
            except:
                print(f"Error spawning vehicle {vehicle_bp} at {spawn_point}")
        _ = world.tick()
        tracks = []
        print('number of vehicles: ', len(world.get_actors().filter("vehicle.*")))
        for i in tqdm(range(n_steps)):
            _ = world.tick()            
            for act in world.get_actors().filter("vehicle.*"):
                # act = world.get_actors().filter("vehicle.*")[0]
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
                till_lane_end_wps = world_map.get_waypoint(act.get_location()).next_until_lane_end(0.25)
                from_lane_start_wps = world_map.get_waypoint(act.get_location()).previous_until_lane_start(0.25)
                if len(from_lane_start_wps) > 1:
                    # No idea why, but cutting of the last waypoint here looks like to be necessary.
                    length_from_lane_start = sum(
                        [
                            from_lane_start_wps[i].transform.location.distance(from_lane_start_wps[i + 1].transform.location)
                            for i in range(len(from_lane_start_wps) - 2)
                        ]
                    )
                else:
                    length_from_lane_start = 0
                if len(till_lane_end_wps) > 1:
                    length_till_lane_end = sum(
                        [
                            till_lane_end_wps[i].transform.location.distance(till_lane_end_wps[i + 1].transform.location)
                            for i in range(len(till_lane_end_wps) - 1)
                        ]
                    )
                else:
                    length_till_lane_end = 0
                row["distance_till_lane_end"] = length_till_lane_end
                row["distance_from_lane_start"] = length_from_lane_start
                row["t_per_lane_id"] = get_t_coordinate(act, world_map)
                row["lane_id"] = world_map.get_waypoint(act.get_location()).lane_id
                row["road_id"] = world_map.get_waypoint(act.get_location()).road_id
                row["carla_road_s"] = world_map.get_waypoint(act.get_location()).s
                row["light_state"] = act.get_light_state()
                row["timestamp"] = world.get_snapshot().timestamp.elapsed_seconds
                forward_vector = act.get_transform().get_forward_vector()
                row["actor_heading_xyz"] = [
                    act.get_transform().get_forward_vector().x,
                    act.get_transform().get_forward_vector().y,
                    act.get_transform().get_forward_vector().z,
                ]
                # Compute speed (dot product of velocity and forward vector)
                speed_lon = (
                    act.get_velocity().x * forward_vector.x
                    + act.get_velocity().y * forward_vector.y
                    + act.get_velocity().z * forward_vector.z
                )
                # Compute acceleration (dot product of acceleration and forward vector)
                acceleration_lon = (
                    act.get_acceleration().x * forward_vector.x
                    + act.get_acceleration().y * forward_vector.y
                    + act.get_acceleration().z * forward_vector.z
                )
                # add lateral speed and accel data
                row["actor_speed_lon"] = speed_lon
                row["actor_acceleration_lon"] = acceleration_lon
                lat_vector = act.get_transform().get_right_vector()
                speed_lat = (
                    act.get_velocity().x * lat_vector.x
                    + act.get_velocity().y * lat_vector.y
                    + act.get_velocity().z * lat_vector.z
                )
                # Compute acceleration (dot product of acceleration and forward vector)
                acceleration_lat = (
                    act.get_acceleration().x * lat_vector.x
                    + act.get_acceleration().y * lat_vector.y
                    + act.get_acceleration().z * lat_vector.z
                )
                row["actor_speed_lat"] = speed_lat
                row["actor_acceleration_lat"] = acceleration_lat
                row["lane_width"] = world_map.get_waypoint(act.get_location()).lane_width
                row["lane_type"] = world_map.get_waypoint(act.get_location()).lane_type
                # Add behavior tracking data
                row["behavior_type"] = vehicle_behaviors.get(act.id, 'normal')
                row["is_slowing_down"] = vehicle_slowdown_timers.get(act.id, 0) > 0
                row["slowdown_remaining_steps"] = vehicle_slowdown_timers.get(act.id, 0)
                
                # Determine vehicle category from type_id
                vehicle_type_id = act.type_id.lower()
                if any(truck_type in vehicle_type_id for truck_type in ['truck', 'van', 'caravan']):
                    row["vehicle_category"] = 'truck'
                elif any(bike_type in vehicle_type_id for bike_type in ['bike', 'motorcycle', 'yamaha', 'kawasaki']):
                    row["vehicle_category"] = 'motorcycle'
                else:
                    row["vehicle_category"] = 'normal'            
                tracks.append(row)
        tracks_df = pd.DataFrame(tracks)
        tracks_df["map"] = world.get_map().name
        tracks_df["scene_id"] = dt_specifier
        tracks_df.to_parquet(f"carla/data/scene_{scene_name}_tracks.parquet")
        print(f"scene {scene_name} saved")
        time.sleep(5)

