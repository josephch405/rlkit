import argparse
import torch

from rlkit.core import logger
from rlkit.samplers.rollout_functions import rollout
from rlkit.envs.wrappers import NormalizedBoxEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.util.gibson import add_env_args, get_config_file, load_env

def simulate_policy(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']

    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.cuda()
        print("set gpu")
    print(ptu.device)

    config_file = get_config_file(args.config_file)
    env = NormalizedBoxEnv(
        load_env(args, config_file, args.env_mode, ptu.device.index))

    print("Policy loaded")
    
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=False,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')

    add_env_args(parser)

    args = parser.parse_args()

    simulate_policy(args)
