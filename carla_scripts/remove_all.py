import carla

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

# Get all actors in the world
actors = world.get_actors()

# Filter out only vehicles, walkers, and sensors
vehicles = actors.filter("vehicle.*")
walkers = actors.filter("walker.*")
sensors = actors.filter("sensor.*")

# Destroy all actors
for actor_list in [vehicles, walkers, sensors]:
    for actor in actor_list:
        actor.destroy()

print("All actors removed!")

