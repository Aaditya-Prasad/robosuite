"""
Record trajectory data with the DataCollectionWrapper wrapper and play them back.

Example:
    $ python demo_collect_and_playback_data.py --environment Lift
"""

import argparse
import os
from glob import glob

import numpy as np

import robosuite as suite
from robosuite.wrappers import DataCollectionWrapper
from robosuite import load_controller_config



def collect_sampled_trajectory(env, timesteps=1000):

    env.reset()
    last_action = np.zeros(env.action_dim)

    for t in range(timesteps):
        x = np.random.uniform(-0.011, 0.011)
        y = np.random.uniform(-0.011, 0.011)
        z = np.random.uniform(-0.06, 0.2)
        translation = np.array([x, y, z])

        theta_deg = np.random.uniform(-90, 90)  # sample rotation angle in degrees
        theta_rad = np.deg2rad(theta_deg)  # convert to radians
        rotation = np.array([0, 0, theta_rad]) 

        action = np.concatenate([translation, rotation, [0.0]]) - last_action
        env.step(action)
        env.render()
        
        last_action = action
        

def playback_trajectory(env, ep_dir):
    """Playback data from an episode.

    Args:
        env (MujocoEnv): environment instance to playback trajectory in
        ep_dir (str): The path to the directory containing data for an episode.
    """

    # first reload the model from the xml
    xml_path = os.path.join(ep_dir, "model.xml")
    with open(xml_path, "r") as f:
        env.reset_from_xml_string(f.read())

    state_paths = os.path.join(ep_dir, "state_*.npz")

    # read states back, load them one by one, and render
    t = 0
    for state_file in sorted(glob(state_paths)):
        print(state_file)
        dic = np.load(state_file)
        states = dic["states"]
        for state in states:
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env.render()
            t += 1
            if t % 100 == 0:
                print(t)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Door")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--directory", type=str, default="/tmp/")
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    # create original environment
    env = suite.make(
        args.environment,
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        robots=args.robots,
        ignore_done=True,
        use_camera_obs=False,
        has_renderer=True,
        has_offscreen_renderer=False,
        control_freq=20,
    )
    data_directory = args.directory

    # wrap the environment with data collection wrapper
    env = DataCollectionWrapper(env, data_directory)

    # testing to make sure multiple env.reset calls don't create multiple directories
    env.reset()
    env.reset()
    env.reset()

    # collect some data
    print("Collecting some random data...")
    collect_sampled_trajectory(env, timesteps=args.timesteps)

    # playback some data
    _ = input("Press any key to begin the playback...")
    print("Playing back the data...")
    data_directory = env.ep_directory
    playback_trajectory(env, data_directory)
