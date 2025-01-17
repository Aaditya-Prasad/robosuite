
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
    des_ee_pos = np.array([0.0, 0.0, 0.9])
    des_ee_euler = np.array([np.pi, 0.0, 0.5])
    des_ee_quat = mat2quat(euler2mat(des_ee_euler))
    des_grip_aa = ee_euler_to_gripper_axisangle(des_ee_euler)

    for i in range(50):
        # obs, _, _, _ = env.step(np.array([1, 1, 1, 1, 1, 1, 0, 0, -10, 0, 0, 0, 1]))
        obs, _, _, _ = env.step(np.concatenate([des_ee_pos, des_grip_aa, np.array([1.0])]))

        ee_pos = obs['robot0_eef_pos']
        ee_quat = obs['robot0_eef_quat']
        ee_euler = mat2euler(quat2mat(ee_quat))
        # print(f"obs['robot0_eef_pos'] = {ee_pos}")
        # print(f"obs['robot0_eef_quat'] = {ee_quat}")
        print(f"obs['robot0_eef_euler'] = {ee_euler[-1]}")

        env.render()

    print(f"reached first position")

    new_des_ee_pos = np.array([-0.1, 0.1, 0.9])
    new_des_ee_euler = np.array([np.pi, 0.0, -0.5])
    new_des_ee_quat = mat2quat(euler2mat(new_des_ee_euler))
    new_des_grip_aa = ee_euler_to_gripper_axisangle(new_des_ee_euler)
    for _ in range(50):
        obs, _, _, _ = env.step(np.concatenate([new_des_ee_pos, new_des_grip_aa, np.array([1.0])]))
        env.render()

    print(f"reached second position")

    def get_e_r(cur_ee_pos, cur_ee_quat, des_ee_pos, des_ee_quat) -> float:
        # get current euler z (after zeroing out x and y)
        cur_ee_euler = mat2euler(quat2mat(cur_ee_quat))
        cur_ee_euler = np.array([0.0, 0.0, cur_ee_euler[-1]])
        # get desired euler z (after zeroing out x and y)
        des_ee_euler = mat2euler(quat2mat(des_ee_quat))
        des_ee_euler = np.array([0.0, 0.0, des_ee_euler[-1]])
        # get error in euler z; make sure to wrap it around correctly
        e_r = des_ee_euler - cur_ee_euler
        if e_r[-1] > np.pi:
            e_r[-1] -= 2 * np.pi
        elif e_r[-1] < -np.pi:
            e_r[-1] += 2 * np.pi
        return e_r

    def get_e_xy(cur_ee_pos, cur_ee_quat, des_ee_pos, des_ee_quat) -> np.ndarray:
        return des_ee_pos - cur_ee_pos

    def get_action(e_r, e_xy, cur_ee_pos, curr_ee_quat) -> np.ndarray:
        des_ee_pos = cur_ee_pos + e_xy
        cur_ee_mat = quat2mat(curr_ee_quat)
        des_ee_mat = euler2mat(e_r) @ cur_ee_mat
        des_ee_euler = mat2euler(des_ee_mat)
        des_grip_aa = ee_euler_to_gripper_axisangle(des_ee_euler)

        return np.concatenate([des_ee_pos, des_grip_aa, np.array([1.0])])

    # now, the goal pose comes in the form of a delta from the current pose
    # i.e. it might e_r w.r.t to current pose
    print(f"heading back to {des_ee_pos}")
    for i in range(50):
        cur_ee_pos = obs['robot0_eef_pos']
        cur_ee_quat = obs['robot0_eef_quat']
        e_r = get_e_r(cur_ee_pos, cur_ee_quat, des_ee_pos, des_ee_quat)
        e_xy = get_e_xy(cur_ee_pos, cur_ee_quat, des_ee_pos, des_ee_quat)
        action = get_action(e_r, e_xy, cur_ee_pos, cur_ee_quat)
        obs, _, _, _ = env.step(action)
        env.render()

    env.step(np.array([1, 1, 1, 1, 1, 1, 0, 0, 10, 0, 0, 0, 1]))

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
        for i in range(1):
            env.step(action)
            env.render()





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

import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--directory", type=str, default="data/")
    parser.add_argument("-t", "--timesteps", type=int, default=15)
    parser.add_argument("-c", "--controller", type=str, default="IK_POSE")
    args = parser.parse_args()

    cont = None

    if args.controller == "OSC_POSE":
        cont = load_controller_config(default_controller="OSC_POSE")
        # cont['impedance_mode'] = "variable_kp"
        cont['impedance_mode'] = "fixed"
        cont['control_delta'] = False
    
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
        robot_eef_init_randomization = True,
        ##these should all be 3-tuples
        robot_eef_pos_min = [-0.5, 0.1, 0.0],
        robot_eef_pos_max = [-0.5, 0.1, 0.0],
        robot_eef_rot_min = [0.0, np.pi/3, 0.0],
        robot_eef_rot_max = [0.0, np.pi/3, 0.0],
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
    env.render()
    time.sleep(100)
    collect_sampled_trajectory(env, timesteps=args.timesteps)

    # playback some data
    data_directory = env.ep_directory
    print(data_directory)
    gather_demonstrations_as_hdf5("data/", "data/", env_info)