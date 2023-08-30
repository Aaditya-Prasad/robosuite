
import argparse
import os
from glob import glob

import numpy as np

import robosuite as suite
from robosuite.wrappers import DataCollectionWrapper
from robosuite import load_controller_config

from PIL import Image

import datetime
import time
import h5py
import json
from robosuite.utils.transform_utils import mat2euler, quat2mat, euler2mat, get_orientation_error, mat2quat, quat2axisangle



def ee_euler_to_gripper_axisangle(des_euler: np.ndarray, rotmat = False) -> np.ndarray:
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

    #just return the rotmatrix
    if rotmat:
        return X_W_grip
    
    #convert to axis angle
    grip_aa = quat2axisangle(mat2quat(X_W_grip))

    return grip_aa

def get_e_r(cur_ee_quat, des_ee_quat) -> float:
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

def get_dpos(cur_ee_pos, des_ee_pos) -> np.ndarray:
    return des_ee_pos - cur_ee_pos



def teleport(env, cur_ee_pos, des_ee_pos, delta_rot, dpos = None):
    """
    Teleports the robot to a specific position. 
    Uses a delta rotation as well as either absolute current/desired pos or delta pos

    Args: 
        env: an instantiated env you want to change
        cur_ee_pos: current ee pose from obs.
        des_ee_pos: desired ee pose.
        delta_rot: delta rotation in euler angles
        dpos: delta position. If you pass this in, cur_ee_pos and des_ee_pos can be None, tp will just use dpos to move the robot
    """
    
    if dpos is None:
        dpos = get_dpos(cur_ee_pos, des_ee_pos)

    drot = euler2mat(delta_rot)

    
    robot = env.robots[0]
    controller = robot.controller
    controller.converge_steps=100

    jpos = controller.joint_positions_for_eef_command(dpos, drot, update_targets=True)
    robot.set_robot_joint_positions(jpos)

    observations = env._get_observations(force_update=True)

    return observations


def move_and_report(env, obs, des_ee_pos, delta_rot, dpos, steps=1):

    for i in range(steps):
        ee_pos = obs['robot0_eef_pos']
        obs = teleport(env, ee_pos, des_ee_pos, delta_rot, dpos = dpos)

    ee_pos = obs['robot0_eef_pos']
    ee_quat = obs['robot0_eef_quat']
    ee_euler = mat2euler(quat2mat(ee_quat))
    print(ee_pos)
    print(ee_euler)
    env.render()

    return obs


def collect_sampled_trajectory(env, dire, timesteps=1000):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    obs = env.reset()
    env.render()

    original_ee_pos = np.array([0.1, -0.25, 1.1])
    ee_pos = obs['robot0_eef_pos']
    ee_quat = obs['robot0_eef_quat']
    ee_euler = mat2euler(quat2mat(ee_quat))

    print("STARTING POSITION")
    print(ee_pos)
    print(ee_euler)

    f = open(dire + "out.csv", "w")

    obs = move_and_report(env, obs, original_ee_pos, delta_rot = np.array([0.0, 0.0, 0.0]), dpos = None, steps=3)
    ## leaving this as a reminder that x and y rotation positions are swapped for some reason
    # obs = move_and_report(env, obs, des_ee_pos, delta_rot = np.array([0.128, 0.0, 0.0]), dpos = None)
    obs_image = obs["robot0_eye_in_hand_image"]
    Image.fromarray(obs_image).save(dire + "images/0.png")
    ee_pos = obs['robot0_eef_pos']
    ee_quat = obs['robot0_eef_quat']
    ee_euler = mat2euler(quat2mat(ee_quat))
    f.write(str(0) + "," + str(ee_pos[0]) + "," + str(ee_pos[1]) + "," + str(ee_pos[2]) + "," + str(ee_quat[0]) + "," + str(ee_quat[1]) + "," + str(ee_quat[2]) + "," + str(ee_quat[3]) + "\n")
    print("AT DEFAULT POSITION")

    for i in range(timesteps):
        x = np.random.uniform(-0.07, 0.02)
        y = np.random.uniform(-0.07, 0.07)
        z = np.random.uniform(-0.1, 0.1) 
        des_ee_pos = np.array([x, y, z]) + original_ee_pos
        obs = move_and_report(env, obs, des_ee_pos, delta_rot = np.array([0.0, 0.0, 0.0]), dpos = None)

        obs_image = obs["robot0_eye_in_hand_image"]
        ee_pos = obs['robot0_eef_pos']
        ee_quat = obs['robot0_eef_quat']
        ee_euler = mat2euler(quat2mat(ee_quat))
        Image.fromarray(obs_image).save(dire + f"images/{i+1}.png")
        f.write(str(i+1) + "," + str(ee_pos[0]) + "," + str(ee_pos[1]) + "," + str(ee_pos[2]) + "," + str(ee_quat[0]) + "," + str(ee_quat[1]) + "," + str(ee_quat[2]) + "," + str(ee_quat[3])  + "\n")


    f.close()





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
    parser.add_argument("-c", "--controller", type=str, default="IK_POSE")
    args = parser.parse_args()

    cont = None

    if args.controller == "OSC_POSE":
        cont = load_controller_config(default_controller="OSC_POSE")
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
        use_camera_obs=True,
        has_renderer=True,
        has_offscreen_renderer=True,
        camera_names="robot0_eye_in_hand",
        camera_heights=128,
        camera_widths=128,
        control_freq=20,
    )

    env_info = json.dumps(config)

    data_directory = args.directory



    # testing to make sure multiple env.reset calls don't create multiple directories
    env.reset()
    env.reset()
    env.reset()

    # collect some data
    print("Collecting some random data...")

    collect_sampled_trajectory(env, data_directory, timesteps=args.timesteps)
