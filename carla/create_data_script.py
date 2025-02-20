import pandas as pd
from src.generate_traffic_data import clean_carla, spawn_scene, run_scene
import carla
import os
os.getcwd()
from datetime import datetime

client = carla.Client('localhost', 2000)

# The following loop runs quite unstable. Hence should be babysitted and restarted if needed.
n_tracks = 10
for i in range(n_tracks):
    print(f'{i} scene from {n_tracks}:')
    world, tm = spawn_scene(8, client, seed = i, sync_mode = False)
    tracks_df = run_scene(world, 50)
    tracks_df['scene_id'] = i
    tracks_df.to_parquet(f"data/scene_{i}_{str(datetime.now().date())}.parquet")
    clean_carla(world)



