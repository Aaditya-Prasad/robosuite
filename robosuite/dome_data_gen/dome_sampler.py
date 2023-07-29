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

import datetime
import h5py
import json



def collect_sampled_trajectory(env, timesteps=1000):

    env.reset()
    env.render()
    stiffness = np.array([1000, 1000, 1000, 100, 100, 0])
    last_action = np.zeros(env.action_dim)
    for i in range(30):
        env.step(np.array([0, 0, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        env.render()
    env.step(np.array([0, 0, 10,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    env.render()
    for t in range(timesteps):
        x = np.random.uniform(-0.011, 0.011)
        y = np.random.uniform(-0.011, 0.011)
        z = np.random.uniform(-0.06, 0.2)
        translation = np.array([x, y, z])

        theta_deg = np.random.uniform(-90, 90)  # sample rotation angle in degrees
        theta_rad = np.deg2rad(theta_deg)  # convert to radians
        rotation = np.array([0, 0, theta_rad]) 
        action = np.concatenate([translation, rotation, stiffness, [0.0]]) - last_action
        for i in range(5):
            env.step(action)
            env.render()
        
        last_action = action
        

def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            print(str(dic["env"]))
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0:
            continue

        # Add only the successful demonstration to dataset
        print("Demonstration is being saved")
            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions), f"OH NO! {len(states)} != {len(actions)}"

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    print(env_name)
    #print env name dtype
    print(type(env_name))
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--directory", type=str, default="data/")
    parser.add_argument("--timesteps", type=int, default=100)
    args = parser.parse_args()

    cont = load_controller_config(default_controller="OSC_POSE")
    cont['impedance_mode'] = "variable_kp"
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": cont,
    }
    # create original environment
    env = suite.make(
        **config,
        ignore_done=True,
        use_camera_obs=False,
        has_renderer=True,
        has_offscreen_renderer=False,
        control_freq=20,
    )

    env_info = json.dumps(config)

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
    data_directory = env.ep_directory
    print(data_directory)
    gather_demonstrations_as_hdf5("data/", "data/", env_info)
