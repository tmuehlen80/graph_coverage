import carla
import random
import time

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

# Get map and spawn points
map = world.get_map()
spawn_points = world.get_map().get_spawn_points()


def find_straight_road_spawn(min_distance=200.0, yaw_tolerance=10):
    """
    Finds a spawn point where the next `min_distance` meters is relatively straight.
    Increases yaw tolerance to allow minor road curves.
    """
    for spawn_point in spawn_points:
        ego_waypoint = map.get_waypoint(spawn_point.location)

        # Get waypoints ahead within min_distance
        waypoints_ahead = ego_waypoint.next(min_distance)
        if len(waypoints_ahead) == 0:
            continue  # Skip if no valid waypoint found

        # Ensure the road remains mostly straight
        is_straight = all(
            abs(waypoint.transform.rotation.yaw - ego_waypoint.transform.rotation.yaw) < yaw_tolerance
            for waypoint in waypoints_ahead
        )

        if is_straight:
            return spawn_point  # Return this spawn point if it's a good match

    return None  # No valid straight road found


# Try to find a straight road spawn point
ego_spawn_point = find_straight_road_spawn()

if ego_spawn_point is None:
    print("⚠ No perfect straight road found! Using default spawn.")
    ego_spawn_point = spawn_points[5]  # Use default spawn if no straight road is found

# Spawn Ego Vehicle
blueprint_library = world.get_blueprint_library()
ego_vehicle_bp = random.choice(blueprint_library.filter("vehicle.mercedes.coupe_2020"))
ego_vehicle = world.spawn_actor(ego_vehicle_bp, ego_spawn_point)
ego_vehicle.set_autopilot(True)
print(f"✅ Ego vehicle spawned at: {ego_spawn_point.location}")


BROKEN_DOWN_DISTANCE = 50.0
broken_location = ego_spawn_point.location + carla.Location(y=0, x=BROKEN_DOWN_DISTANCE, z=0)
broken_down_waypoint = map.get_waypoint(broken_location)

# Ensure correct lane positioning
broken_spawn_point = broken_down_waypoint.transform
broken_spawn_point.location.z += 0.5  # Prevent underground spawns

# Check for obstacles
actors_nearby = world.get_actors().filter("vehicle.*")
safe_to_spawn = all(actor.get_location().distance(broken_spawn_point.location) > 5.0 for actor in actors_nearby)

if safe_to_spawn:
    broken_vehicle_bp = random.choice(blueprint_library.filter("vehicle.nissan.micra"))
    broken_vehicle = world.spawn_actor(broken_vehicle_bp, broken_spawn_point)
    broken_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    print(f"✅ Broken-down vehicle spawned at {broken_spawn_point.location}")
else:
    print("⚠ Collision risk detected! Skipping broken-down vehicle spawn.")

oncoming_waypoint = broken_down_waypoint.get_left_lane()  # Find opposite lane
if oncoming_waypoint:
    oncoming_location = ego_spawn_point.location + carla.Location(x=0, y=-100, z=0)  # Place 100m behind
    oncoming_waypoint = map.get_waypoint(oncoming_location)

    oncoming_spawn_point = oncoming_waypoint.transform
    oncoming_spawn_point.location.z += 0.5  # Prevent underground spawns

    # Check for collisions
    safe_to_spawn = all(actor.get_location().distance(oncoming_spawn_point.location) > 8.0 for actor in actors_nearby)

    if safe_to_spawn:
        oncoming_bp = random.choice(blueprint_library.filter("vehicle.tesla.model3"))
        oncoming_vehicle = world.spawn_actor(oncoming_bp, oncoming_spawn_point)
        oncoming_vehicle.set_autopilot(True)
        print("✅ Oncoming vehicle spawned safely.")
    else:
        print("⚠ Collision risk detected! Skipping oncoming vehicle spawn.")
else:
    print("⚠ No opposite lane found! Skipping oncoming vehicle.")


# spectator = world.get_spectator()
#
# def update_spectator():
#    transform = ego_vehicle.get_transform()
#    camera_location = transform.location + carla.Location(x=-6, y=0, z=3)
#    camera_rotation = carla.Rotation(pitch=-15, yaw=transform.rotation.yaw, roll=0)
#    spectator.set_transform(carla.Transform(camera_location, camera_rotation))
#
## Continuously update spectator camera
# while True:
#    update_spectator()
#    time.sleep(0.1)
#
