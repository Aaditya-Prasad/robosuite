
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
from robosuite.utils.transform_utils import mat2euler, quat2mat, euler2mat, get_orientation_error, mat2quat, quat2axisangle



def ee_euler_to_gripper_axisangle(des_euler: np.ndarray) -> np.ndarray:
    """Converts an euler angle for the end effector to axis angle orientation for the gripper.

    Returns:
        grip_aa: np.ndarray
            Axis angle representation for actions to be passed to env.step().
    """
    # Manual conversion matrix
    X_ee_grip = quat2mat(np.array([0, 0, -0.707107, 0.707107]))

    # Convert desired Euler angles to rotation matrix
    X_W_ee = euler2mat(des_euler)

    # Calculate the combined transformation matrix
    X_W_grip = X_W_ee @ X_ee_grip

    # Convert the combined matrix to axis-angle representation
    grip_aa = quat2axisangle(mat2quat(X_W_grip))

    return grip_aa


def collect_sampled_trajectory(env, timesteps=1000):

    env.reset()
    env.render()
    print("Robot start")
    stiffness = np.array([10, 10, 10, 10, 10, 10])
    des_ee_pos = np.array([0.2, -0.25, 1.15])
    des_ee_euler = np.array([np.pi, 0.0, 0.5])
    des_ee_quat = mat2quat(euler2mat(des_ee_euler))
    des_grip_aa = ee_euler_to_gripper_axisangle(des_ee_euler)

    ycount = 0
    xcount = 0

    for i in range(50):
        # obs, _, _, _ = env.step(np.array([1, 1, 1, 1, 1, 1, 0, 0, -10, 0, 0, 0, 1]))
        obs, _, _, _ = env.step(np.concatenate([des_ee_pos, des_grip_aa, np.array([1.0])]))

        ee_pos = obs['robot0_eef_pos']
        ee_quat = obs['robot0_eef_quat']
        ee_euler = mat2euler(quat2mat(ee_quat))



        env.render()

    print(f"reached first position")

    for i in range(timesteps):
        #x should be +- .14 of 0.0
        #y should be +- .14 of -0.35
        x = np.random.uniform(-0.06, 0.06) + 0.2
        y = np.random.uniform(-0.06, 0.06) - 0.25
        z = np.random.uniform(1.1, 1.15)
        des_ee_pos = np.array([x, y, z])

        # if i % 2 == 0:
        #     des_ee_pos = np.array([0.2, -0.25, 1.15])
        #     des_ee_euler = np.array([np.pi, 0.0, 0.5])

        # if i % 2 == 1:
        #     des_ee_pos = np.array([0.2, -0.25, 1.15])
        #     des_ee_euler = np.array([np.pi, 0.0, 0.5])
        
        for _ in range(75):
            obs, _, _, _ = env.step(np.concatenate([des_ee_pos, des_grip_aa, np.array([1.0])]))

            ee_pos = obs['robot0_eef_pos']
            ee_quat = obs['robot0_eef_quat']
            ee_euler = mat2euler(quat2mat(ee_quat))



            env.render()
        print(f"reached position {i}")
        print(f"y ori = {ee_euler[1]}")
        print(f"x ori = {ee_euler[0]}")
        if np.abs(ee_euler[1]) > 0.01:
            ycount += 1
        if np.abs(ee_euler[0]-np.pi) > 0.01:
            xcount += 1

    print(f"ycount = {ycount}")
    print(f"xcount = {xcount}")






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
    parser.add_argument("--environment", type=str, default="PickPlaceCan")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--directory", type=str, default="can_data/")
    parser.add_argument("-t", "--timesteps", type=int, default=15)
    parser.add_argument("-c", "--controller", type=str, default="OSC_POSE")
    args = parser.parse_args()

    cont = None

    if args.controller == "OSC_POSE":
        cont = load_controller_config(default_controller="OSC_POSE")
        cont['impedance_mode'] = "fixed"
        cont['control_delta'] = False
    
    if args.controller == "IK_POSE":
        raise NotImplementedError("IK_POSE not implemented yet")

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
    gather_demonstrations_as_hdf5("can_data/", "can_data/", env_info)