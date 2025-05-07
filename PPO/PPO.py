import gym
from stable_baselines3 import PPO

env_name = "loaddisEnv-v0"
env = gym.make(env_name)

model = PPO("MlpPolicy",
            env=env,
            learning_rate=3e-4,
            batch_size=2,
            gae_lambda=0.95,
            gamma=0.5,
            n_steps=10,
            verbose=0
)


model.learn(total_timesteps=9000000000)

model.save("./model/loaddec_PPO.pkl")

env = gym.make(env_name)
model = A2C.load("./model/loaddec_PPO.pkl")

state = env.reset()
done = False
score = 0
while not done:
    action, _ = model.predict(observation=state)
    state, reward, done, info = env.step(action=action)
    score += reward
env.close()
score