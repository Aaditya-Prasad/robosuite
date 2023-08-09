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
from robosuite.utils.transform_utils import quat2mat, euler2mat, get_orientation_error, mat2quat

def test(env):
    env.reset()
    env.render()
    print("Testing")
    stiffness = np.array([10, 10, 10, 10, 10, 10])
    for i in range(300):
        env.step(np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]))
        env.render()
    print("Done turning")
    for i in range(300):
        env.step(np.array([1, 1, 1, 1, 1, 1, 10, 0, 0, 0, 0, 0, 1]))
        env.render()

def collect_sampled_trajectory(env, timesteps=1000):

    env.reset()
    env.render()
    print("Robot start")
    stiffness = np.array([10, 10, 10, 10, 10, 10])
    last_action = np.zeros(env.action_dim)
    for i in range(300):
        env.step(np.array([1, 1, 1, 1, 1, 1, 0, 0, -10, 0, 0, 0, 1]))
        env.render()
    env.step(np.array([1, 1, 1, 1, 1, 1, 0, 0, 10, 0, 0, 0, 1]))
    print("Robot in position")
    env.render()
    for t in range(timesteps):
        x = np.random.uniform(0, 0.011)
        y = np.random.uniform(-0.011, 0.011) 
        z = np.random.uniform(-0.06, 0.2)
        translation = np.array([x, y, z])

        theta_deg = np.random.uniform(-90, 90)  # sample rotation angle in degrees
        theta_rad = np.deg2rad(theta_deg)  # convert to radians
        rotation = np.array([0, 0, theta_rad]) 
        action = np.concatenate([stiffness, translation, rotation, [1.0]])
        # action = np.concatenate([translation, rotation, [0.0]]) - last_action
        for i in range(1):
            env.step(action)
            env.render()
        
        last_action = action


def calc_action(observation, target_pos, target_ori, scale = 10):
    cur_ori = observation['robot0_eef_quat']
    cur_pos = observation['robot0_eef_pos']
    cur_mat = quat2mat(cur_ori)
    cur_mat = cur_mat @ euler2mat(np.array([0, 0, -np.pi/2]))
    
    pos_dif = target_pos - cur_pos
    rot_dif= np.matmul(cur_mat.T, get_orientation_error(target_ori, cur_ori))
    action = np.hstack([pos_dif/scale, rot_dif, np.array([1])])  # last number is for the gripper
    return action

def check(observation, target_pos, target_ori, tolerance=0.02):
    cur_ori = observation['robot0_eef_quat']
    cur_pos = observation['robot0_eef_pos']
    cur_mat = quat2mat(cur_ori)
    cur_mat = cur_mat @ euler2mat(np.array([0, 0, -np.pi/2]))
    
    pos_dif = target_pos - cur_pos
    rot_dif= np.matmul(cur_mat.T, get_orientation_error(target_ori, cur_ori))

    if np.linalg.norm(pos_dif) < tolerance and np.linalg.norm(rot_dif) < tolerance:
        return True
    return False    

def collect_ik_trajectory(env, timesteps=1000):
        obs = env.reset()
        print("CURR_ORI", obs['robot0_eef_quat'])
        print("CURR_POS", obs['robot0_eef_pos'])
        env.render()
        print("Robot start")
        while True:
            action = calc_action(obs, np.array([0.0, 0.0, .85]), np.array([1.0, 0.0, 0.0, 0.0]))
            obs, _, _, _ = env.step(action)
            env.render()
            if check(obs, np.array([0.0, 0.0, .85]), np.array([1.0, 0.0, 0.0, 0.0]), tolerance=0.01):
                print("Robot in position")
                break
        for t in range(timesteps):
            x = np.random.uniform(0, 0.011)
            y = np.random.uniform(-0.011, 0.011) 
            z = np.random.uniform(0.02, 0.04)
            translation = np.array([x, y, z])

            theta_deg = np.random.uniform(-90, 90)
            theta_rad = np.deg2rad(theta_deg)
            rotation = np.array([0, 0, theta_rad])

            desired_pos = obs['robot0_eef_pos'] + translation
            desired_ori = mat2quat(quat2mat(obs['robot0_eef_quat']) @ euler2mat(rotation))
            desired_ori[0] = 1.0
            
            print("DESIRED_ORI", np.round(desired_ori, 2), "DESIRED_POS", np.round(desired_pos, 2))
            print("CURR_ORI", np.round(obs['robot0_eef_quat'], 2), "CURR_POS", np.round(obs['robot0_eef_pos'], 2))

            i = 0
            while True:
                action = calc_action(obs, desired_pos, desired_ori, scale = 20)
                obs, _, _, _ = env.step(action)
                i += 1
                env.render()
                if i % 100 == 0:
                    print("CURR_ORI", np.round(obs['robot0_eef_quat'], 2))
                    print("CURR_POS", np.round(obs['robot0_eef_pos'], 2))
                if check(obs, desired_pos, desired_ori):
                    print("Robot reached position")
                    break
            

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

    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--directory", type=str, default="data/")
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("-c", "--controller", type=str, default="OSC_POSE")
    args = parser.parse_args()

    cont = None

    if args.controller == "OSC_POSE":
        cont = load_controller_config(default_controller="OSC_POSE")
        cont['impedance_mode'] = "variable_kp"
    
    if args.controller == "IK_POSE":
        cont = load_controller_config(default_controller="IK_POSE")

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
    # test(env)
    if args.controller == "IK_POSE":
        collect_ik_trajectory(env, timesteps=args.timesteps)
    else:
        collect_sampled_trajectory(env, timesteps=args.timesteps)

    # playback some data
    data_directory = env.ep_directory
    print(data_directory)
    gather_demonstrations_as_hdf5("data/", "data/", env_info)
