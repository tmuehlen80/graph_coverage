import carla
import random
import time

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

# Get blueprint library
blueprint_library = world.get_blueprint_library()

# Choose a random vehicle
vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))

# Get a random spawn point
spawn_point = random.choice(world.get_map().get_spawn_points())

# Spawn the vehicle
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
print(f"Spawned vehicle at: {spawn_point.location}")

# Move the spectator camera to follow the vehicle
spectator = world.get_spectator()


# Function to update spectator view
def update_spectator():
    transform = vehicle.get_transform()  # Get vehicle position
    location = transform.location + carla.Location(z=5)
    rotation = carla.Rotation(pitch=-90, yaw=transform.rotation.yaw, roll=0)  # Slight downward tilt
    spectator.set_transform(carla.Transform(location, rotation))


# Initial camera placement
update_spectator()

vehicle.set_autopilot(True)
# Keep updating the camera for 10 seconds
start_time = time.time()
while time.time() - start_time < 10:
    update_spectator()
    time.sleep(0.1)

# Enable autopilot


print("Vehicle spawned and spectator camera attached!")
