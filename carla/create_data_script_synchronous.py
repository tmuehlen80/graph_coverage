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


# BEHAVIOR CONTROL OPTIONS
# These parameters control actor behavior variability to create more diverse traffic scenarios
# All boolean flags can be set to False to disable specific behavior variations
# Speed variation parameters
SPEED_VARIATION_ENABLED = True
MIN_SPEED_FACTOR = 0.6  # 60% of speed limit
MAX_SPEED_FACTOR = 1.4  # 140% of speed limit
# Distance variation parameters 
DISTANCE_VARIATION_ENABLED = True
MIN_DISTANCE_FACTOR = 0.5  # Closer following distance
MAX_DISTANCE_FACTOR = 3.0  # Larger following distance
# Aggressive/Conservative driving parameters
AGGRESSIVE_DRIVING_ENABLED = True
AGGRESSIVE_PROBABILITY = 0.3  # 30% of vehicles drive aggressively
# Lane change behavior parameters
LANE_CHANGE_VARIATION_ENABLED = True
MIN_LANE_CHANGE_DISTANCE = 10.0  # Meters
MAX_LANE_CHANGE_DISTANCE = 50.0  # Meters
# Braking behavior parameters
BRAKING_VARIATION_ENABLED = True
MIN_BRAKING_DISTANCE = 5.0  # Meters
MAX_BRAKING_DISTANCE = 15.0  # Meters
# Random behavior parameters
RANDOM_SLOW_DOWNS_ENABLED = True
SLOW_DOWN_PROBABILITY = 0.1  # 10% chance per step for random slowdown
SLOW_DOWN_DURATION_RANGE = (50, 200)  # Steps to maintain slow speed
# Vehicle type variation (affects behavior)
VEHICLE_TYPE_VARIATION_ENABLED = True
TRUCK_PROBABILITY = 0.2  # 20% trucks (slower, different following distance)
MOTORCYCLE_PROBABILITY = 0.1  # 10% motorcycles (faster, more aggressive)
# Traffic light behavior variation
TRAFFIC_LIGHT_VARIATION_ENABLED = True
EARLY_BRAKE_PROBABILITY = 0.3  # 30% brake early at yellow lights
LATE_BRAKE_PROBABILITY = 0.2   # 20% brake late at yellow lights
# Lane changing behavior variation
FREQUENT_LANE_CHANGES_ENABLED = True
FREQUENT_CHANGER_PROBABILITY = 0.15  # 15% of vehicles change lanes frequently
# Weather/visibility response (can be used with weather variations)
WEATHER_RESPONSE_ENABLED = True
CAUTIOUS_IN_WEATHER_PROBABILITY = 0.6  # 60% drive more cautiously in bad weather
# Get the traffic manager


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
        tm = client.get_trafficmanager(8000)  # Port 8000
        tm.set_synchronous_mode(True)  # Make TM sync with simulation
        # Configure global traffic manager behavior for variation
        # if SPEED_VARIATION_ENABLED: # throws error
        #     # tm.set_global_percentage_speed_difference(-20.0)  # Base speed reduction
        #     tm.vehicle_percentage_speed_difference(-20.0)  # Base speed reduction
        if DISTANCE_VARIATION_ENABLED:
            tm.set_global_distance_to_leading_vehicle(2.0)  # Base following distance
        # Get blueprint library
        blueprint_library = world.get_blueprint_library()
        # Spawn vehicles
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        # Track vehicle behavior states
        vehicle_behaviors = {}
        vehicle_slowdown_timers = {}
        n_vehicles = random.randint(10, 60)
        for i in range(n_vehicles):
            spawn_point = spawn_points[i]
            # Select vehicle type based on probabilities
            if VEHICLE_TYPE_VARIATION_ENABLED:
                rand_val = random.random()
                if rand_val < TRUCK_PROBABILITY:
                    # Spawn truck/larger vehicle
                    truck_bps = [bp for bp in blueprint_library.filter("vehicle.*") if 
                                any(truck_type in bp.id.lower() for truck_type in ['truck', 'van', 'caravan'])]
                    if truck_bps:
                        vehicle_bp = random.choice(truck_bps)
                        vehicle_type = 'truck'
                    else:
                        vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
                        vehicle_type = 'normal'
                elif rand_val < TRUCK_PROBABILITY + MOTORCYCLE_PROBABILITY:
                    # Spawn motorcycle/bike
                    bike_bps = [bp for bp in blueprint_library.filter("vehicle.*") if 
                            any(bike_type in bp.id.lower() for bike_type in ['bike', 'motorcycle', 'yamaha', 'kawasaki'])]
                    if bike_bps:
                        vehicle_bp = random.choice(bike_bps)
                        vehicle_type = 'motorcycle'
                    else:
                        vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
                        vehicle_type = 'normal'
                else:
                    # Spawn regular car
                    vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
                    vehicle_type = 'normal'
            else:
                vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
                vehicle_type = 'normal'
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            # Store vehicle reference for behavior control
            vehicle_id = vehicle.id
            # Enable autopilot
            vehicle.set_autopilot(True, tm.get_port())  # TM handles driving
            # Apply vehicle type-specific behaviors
            if vehicle_type == 'truck':
                # Trucks: slower, larger following distance, more conservative
                tm.vehicle_percentage_speed_difference(vehicle, random.uniform(-30.0, -10.0))  # Slower
                tm.distance_to_leading_vehicle(vehicle, random.uniform(3.0, 5.0))  # Larger following distance
                vehicle_behaviors[vehicle_id] = 'truck_conservative'
            elif vehicle_type == 'motorcycle':
                # Motorcycles: faster, smaller following distance, more aggressive
                tm.vehicle_percentage_speed_difference(vehicle, random.uniform(10.0, 40.0))  # Faster
                tm.distance_to_leading_vehicle(vehicle, random.uniform(0.5, 1.5))  # Smaller following distance
                vehicle_behaviors[vehicle_id] = 'motorcycle_aggressive'
            else:
                # Apply normal individual vehicle behavior variations
                if SPEED_VARIATION_ENABLED:
                    # Random speed factor for each vehicle
                    speed_factor = random.uniform(MIN_SPEED_FACTOR, MAX_SPEED_FACTOR)
                    speed_percentage = (speed_factor - 1.0) * 100.0
                    tm.vehicle_percentage_speed_difference(vehicle, speed_percentage)
                if DISTANCE_VARIATION_ENABLED:
                    # Random following distance for each vehicle
                    distance_factor = random.uniform(MIN_DISTANCE_FACTOR, MAX_DISTANCE_FACTOR)
                    tm.distance_to_leading_vehicle(vehicle, distance_factor)
                if AGGRESSIVE_DRIVING_ENABLED:
                    # Some vehicles drive more aggressively
                    if random.random() < AGGRESSIVE_PROBABILITY:
                        tm.vehicle_percentage_speed_difference(vehicle, random.uniform(20.0, 50.0))  # Faster
                        tm.distance_to_leading_vehicle(vehicle, random.uniform(0.5, 1.0))  # Closer following
                        vehicle_behaviors[vehicle_id] = 'aggressive'
                    else:
                        vehicle_behaviors[vehicle_id] = 'conservative'
            if LANE_CHANGE_VARIATION_ENABLED:
                # Vary lane change distances
                lane_change_distance = random.uniform(MIN_LANE_CHANGE_DISTANCE, MAX_LANE_CHANGE_DISTANCE)
                tm.set_desired_speed(vehicle, random.uniform(20, 50))  # km/h variation
            if FREQUENT_LANE_CHANGES_ENABLED:
                # Some vehicles change lanes more frequently
                if random.random() < FREQUENT_CHANGER_PROBABILITY:
                    tm.auto_lane_change(vehicle, True)  # Enable automatic lane changes
                    # Make them more likely to change lanes by reducing lane change distance
                    tm.distance_to_leading_vehicle(vehicle, 1.0)  # Smaller following distance encourages lane changes
                    if vehicle_behaviors.get(vehicle_id) not in ['truck_conservative', 'motorcycle_aggressive']:
                        vehicle_behaviors[vehicle_id] = 'frequent_changer'
            # if BRAKING_VARIATION_ENABLED:
            #     # Set random collision detection distances (affects braking behavior)
            #     collision_distance = random.uniform(MIN_BRAKING_DISTANCE, MAX_BRAKING_DISTANCE)
            #     tm.collision_detection(vehicle, vehicle, vehicle, collision_distance)
            # Initialize random slowdown timer
            if RANDOM_SLOW_DOWNS_ENABLED:
                vehicle_slowdown_timers[vehicle_id] = 0        
            # vehicles.append(vehicle)
        _ = world.tick()
        tracks = []
        print('number of vehicles: ', len(world.get_actors().filter("vehicle.*")))
        for i in tqdm(range(n_steps)):
            _ = world.tick()
            # Apply dynamic behavior changes during simulation
            if RANDOM_SLOW_DOWNS_ENABLED:
                for act in world.get_actors().filter("vehicle.*"):
                    vehicle_id = act.id
                    # Check if vehicle should start slowing down
                    if vehicle_slowdown_timers.get(vehicle_id, 0) == 0:
                        if random.random() < SLOW_DOWN_PROBABILITY:
                            # Start random slowdown
                            slowdown_duration = random.randint(*SLOW_DOWN_DURATION_RANGE)
                            vehicle_slowdown_timers[vehicle_id] = slowdown_duration
                            # Apply temporary speed reduction
                            tm.vehicle_percentage_speed_difference(act, -50.0)  # 50% speed reduction
                    # Update slowdown timer
                    elif vehicle_slowdown_timers[vehicle_id] > 0:
                        vehicle_slowdown_timers[vehicle_id] -= 1
                        # If timer reaches zero, restore normal speed
                        if vehicle_slowdown_timers[vehicle_id] == 0:
                            # Restore original speed behavior
                            if vehicle_behaviors.get(vehicle_id) == 'aggressive':
                                tm.vehicle_percentage_speed_difference(act, random.uniform(20.0, 50.0))
                            else:
                                speed_factor = random.uniform(MIN_SPEED_FACTOR, MAX_SPEED_FACTOR)
                                speed_percentage = (speed_factor - 1.0) * 100.0
                                tm.vehicle_percentage_speed_difference(act, speed_percentage)        
            # Apply periodic behavior variations every 120 steps
            if i % 120 == 0 and i > 0:
                for act in world.get_actors().filter("vehicle.*"):
                    # Randomly change some vehicle behaviors
                    if random.random() < 0.2:  # 20% chance to change behavior
                        if DISTANCE_VARIATION_ENABLED:
                            new_distance = random.uniform(MIN_DISTANCE_FACTOR, MAX_DISTANCE_FACTOR)
                            tm.distance_to_leading_vehicle(act, new_distance)
                        
                        if SPEED_VARIATION_ENABLED and vehicle_slowdown_timers.get(act.id, 0) == 0:
                            # Only change speed if not in slowdown mode
                            speed_factor = random.uniform(MIN_SPEED_FACTOR, MAX_SPEED_FACTOR)
                            speed_percentage = (speed_factor - 1.0) * 100.0
                            tm.vehicle_percentage_speed_difference(act, speed_percentage)
            
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
                # till_lane_end_wps = [world_map.get_waypoint(act.get_location())] + till_lane_end_wps
                from_lane_start_wps = world_map.get_waypoint(act.get_location()).previous_until_lane_start(0.25)
                # from_lane_start_wps.append(world_map.get_waypoint(act.get_location()))
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

