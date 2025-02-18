import pandas as pd
from src.generate_traffic_data import clean_carla, spawn_scene, run_scene
import carla
import os
os.getcwd()

client = carla.Client('localhost', 2000)

world = spawn_scene(5, client, sync_mode = False)

tracks_df = run_scene(world, 1000)

len(world.get_actors().filter("vehicle.*"))

dir(world.get_actors().filter("vehicle.*")[0])


world.get_actors().filter("vehicle.*")[0]
clean_carla(world)
tracks_df.head(20).T

tracks_df.describe()
