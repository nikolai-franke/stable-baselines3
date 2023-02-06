import gym

from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.dsac.dsac import DSAC


def main():
    # env_id = "CartPole-v1"
    env_kwargs = {
        # "obs_type": "ram",
        # "frame_skip": 0,
    }
    # env = make_atari_env("RoadRunner-v4", n_envs=24)
    env = make_atari_env("ALE/MsPacman-v5", n_envs=24, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=4)
    # env = make_vec_env(env_id, n_envs=24, vec_env_cls=SubprocVecEnv)
    policy_kwargs = {
        "net_arch": [512, 512],
    }

    model = DSAC(
        "CnnPolicy",
        # "MlpPolicy",
        env,
        buffer_size=400_000,
        learning_starts=10_000,
        target_update_interval=4,
        train_freq=1,
        verbose=1,
        batch_size=512,
        learning_rate=1e-4,
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=500_000, log_interval=20)


if __name__ == "__main__":
    main()
