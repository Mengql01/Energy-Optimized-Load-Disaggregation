import gym
from stable_baselines3 import A2C

env_name = "loaddisEnv-v0"
env = gym.make(env_name)

model = A2C("MlpPolicy",
            env=env,
            learning_rate=3e-4,
            gae_lambda=0.95,
            gamma=0.5,
            n_steps=10,
            verbose=0
)


model.learn(total_timesteps=9000000000)

model.save("./model/loaddec_A2C.pkl")

env = gym.make(env_name)
model = A2C.load("./model/loaddec_A2C.pkl")

state = env.reset()
done = False
score = 0
while not done:
    action, _ = model.predict(observation=state)
    state, reward, done, info = env.step(action=action)
    score += reward
env.close()
score