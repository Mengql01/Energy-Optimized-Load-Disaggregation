import gym
import numpy as np
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
import os
from typing import List


class SaveDataCallback(BaseCallback):
    def __init__(self, check_interval=100, save_dir="./training_data_(DAY1,10%))", verbose=0):
        super().__init__(verbose)
        self.check_interval = check_interval
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.episode_count = 0
        self.data_buffer = {
            "actions": [],
            "states": [],
            "rewards": []
        }
        self.csv_header_written = False

    def _on_step(self) -> bool:
        self.data_buffer["actions"].append(self.locals["actions"])
        self.data_buffer["states"].append(self.locals["new_obs"])
        self.data_buffer["rewards"].append(self.locals["rewards"])

        done_flags = self.locals["dones"]
        self.episode_count += sum(done_flags)

        if self.episode_count >= self.check_interval:
            self._save_to_csv()
            self.episode_count = 0
        return True

    def _save_to_csv(self):
        filename = f"(DAY1,10%))_{self.num_timesteps}.csv"
        file_path = os.path.join(self.save_dir, filename)
        flattened_data = self._flatten_buffer_data()
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not self.csv_header_written:
                header = self._generate_header()
                writer.writerow(header)
                self.csv_header_written = True
            for row in flattened_data:
                writer.writerow(row)
        self.data_buffer = {k: [] for k in self.data_buffer}

    def _flatten_buffer_data(self) -> List[list]:
        flattened = []
        for i in range(len(self.data_buffer["actions"])):
            for env_idx in range(self.data_buffer["actions"][i].shape[0]):
                row = [
                    self.num_timesteps,
                    self.episode_count,
                    env_idx,
                    *self.data_buffer["actions"][i][env_idx].flatten().tolist(),
                    *self.data_buffer["states"][i][env_idx].flatten().tolist(),
                    self.data_buffer["rewards"][i][env_idx].item()
                ]
                flattened.append(row)
        return flattened

    def _generate_header(self) -> List[str]:
        action_dim = self.data_buffer["actions"][0].shape[-1]
        state_dim = self.data_buffer["states"][0].shape[-1]

        header = [
            "timestep",
            "episode",
            "env_id",
            *[f"action_{i}" for i in range(action_dim)],
            *[f"state_{i}" for i in range(state_dim)],
            "reward"
        ]
        return header


def make_env(env_name):
    return lambda: gym.make(env_name)


if __name__ == '__main__':
    env = SubprocVecEnv([make_env("loaddecEnv-v4") for _ in range(32)])
    env = VecMonitor(env)
    model = PPO("MlpPolicy",
                env=env,
                learning_rate=1e-1,
                batch_size=32,
                gamma=0.5,
                n_steps=10,
                verbose=0,
                # device="cuda",
                tensorboard_log="./tensorboard/loaddecEnv-v4/")


    callback = SaveDataCallback(check_interval=100)
    model.learn(total_timesteps=500000, callback=callback, tb_log_name="(DAY1,10%))")