import os
import gibson2
from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv, InteractiveNavigateEnv

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

def get_config_file(config_file_name):
    return os.path.join(os.path.dirname(
        gibson2.__file__), "../examples/configs", config_file_name)
    