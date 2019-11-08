import os
import argparse

import gibson2
from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv, InteractiveNavigateEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def add_sac_args(parser):
    parser.add_argument(
        "-n",
        "--exp-name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--layer-size",
        type=int,
        default=256
    )
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=int(1E3)
    )


def add_env_args(parser):
    group = parser.add_argument_group("env")
    group.add_argument(
        "--env-type",
        required=True,
        help="env type: [gibson|toy]",
        choices=["gibson", "toy"]
    )
    group.add_argument(
        "--config-file",
        required=True,
        help="config yaml file for Gibson environment",
    )
    group.add_argument(
        "--env-mode",
        type=str,
        default="headless",
        help="environment mode for the simulator (default: headless)",
    )
    group.add_argument(
        "--action-timestep",
        type=float,
        default=1.0 / 10.0,
        help="action timestep for the simulator (default: 0.1)",
    )
    group.add_argument(
        "--physics-timestep",
        type=float,
        default=1.0 / 40.0,
        help="physics timestep for the simulator (default: 0.025)",
    )
    group.add_argument(
        "--random-position",
        action="store_true",
        default=False,
        help="whether to randomize initial and target position (default: False)",
    )
    group.add_argument(
        "--random-height",
        action="store_true",
        default=False,
        help="whether to randomize the height of target position (default: False)",
    )


# config_file
# action_timestep

def load_env(args, config_file, env_mode, device_idx):
    if args.env_type == "gibson":
        if args.random_position:
            return NavigateRandomEnv(config_file=config_file,
                                     mode=env_mode,
                                     action_timestep=args.action_timestep,
                                     physics_timestep=args.physics_timestep,
                                     random_height=args.random_height,
                                     automatic_reset=True,
                                     device_idx=device_idx)
        else:
            return NavigateEnv(config_file=config_file,
                               mode=env_mode,
                               action_timestep=args.action_timestep,
                               physics_timestep=args.physics_timestep,
                               automatic_reset=True,
                               device_idx=device_idx)
    elif args.env_type == "interactive_gibson":
        return InteractiveNavigateEnv(config_file=config_file,
                                      mode=env_mode,
                                      action_timestep=args.action_timestep,
                                      physics_timestep=args.physics_timestep,
                                      automatic_reset=True,
                                      random_position=args.random_position,
                                      device_idx=device_idx)


def experiment(variant, args):

    config_file = os.path.join(os.path.dirname(
        gibson2.__file__), "../examples/configs", args.config_file)

    expl_env = NormalizedBoxEnv(
        load_env(args, config_file, args.env_mode, ptu.device.index))
    eval_env = NormalizedBoxEnv(
        load_env(args, config_file, args.env_mode, ptu.device.index))

    # TODO: read dynamically from config_file
    display_dim = 64

    sensor_dim = 3
    rgb_dim = display_dim * display_dim * 3
    depth_dim = display_dim * display_dim * 1
    obs_dim = sensor_dim + rgb_dim + depth_dim
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker

    parser = argparse.ArgumentParser()
    add_env_args(parser)
    add_sac_args(parser)
    args = parser.parse_args()

    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=args.layer_size,
        replay_buffer_size=args.replay_buffer_size,
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger(args.exp_name, variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant, args)
