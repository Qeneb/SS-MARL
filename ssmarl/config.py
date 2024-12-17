import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description='ssmarl', formatter_class=argparse.RawDescriptionHelpFormatter)

    # train parameters
    parser.add_argument("--algorithm_name", type=str, default='ssmarl', choices=['ssmarl'])
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--device",  type=str, default='cuda:3', help="device: cpu/cuda:X")
    parser.add_argument("--cuda_deterministic", action='store_false', default=True,
                        help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int, default=128,
                        help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=8,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')

    # scenario parameters
    parser.add_argument('--scenario_name', type=str, default='exp2',
                        help="Which scenario to run on")
    parser.add_argument("--num_obstacles", type=int, default=3,
                        help="number of obstacles")
    parser.add_argument('--num_agents', type=int, default=3,
                        help="number of players")
    parser.add_argument('--num_landmarks', type=int, default=3,
                        help="number of landmarks")
    parser.add_argument("--restore_model", type=bool, default=True,
                        help="for fine tunning/rendering")
    parser.add_argument("--parameter_share", type=bool, default=True,
                        help='parameter sharing')

    # wandb parameters
    parser.add_argument("--user_name", type=str, default='431',
                        help="for wandb usage.")
    parser.add_argument("--wandb_project_name", type=str, default='EXP',
                        help="for wandb usage.")
    parser.add_argument("--experiment_name", type=str, default="test",
                        help="an identifier to distinguish different experiment.")
    parser.add_argument("--use_wandb", action='store_false', default=True,
                        help="for wandb usage.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='MPE',
                        help="specify the name of environment")
    parser.add_argument("--episode_length", type=int, default=100,
                        help="Max length for any episode")

    # network parameters
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_popart", action='store_false', default=True,
                        help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True,
                        help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain of ACT layer")
    parser.add_argument("--gnn_num_heads", type=int, default=1,
                        help='head of graph attention')

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true', default=False,
                        help='Default: False')
    parser.add_argument("--use_recurrent_policy", action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1,
                        help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")
    
    # optimizer parameters
    parser.add_argument("--lr", type=float, default=7e-3,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=7e-3,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='Adam optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # trpo parameters
    parser.add_argument("--kl_threshold", type=float, default=0.05,
                        help='the threshold of kl-divergence (default: 0.01)')
    parser.add_argument("--safety_bound", type=float, default=1,
                        help='the upper bound of cost constrain (default: 1)')
    parser.add_argument("--ls_step", type=int, default=10,
                        help='number of line search (default: 10)')
    parser.add_argument("--accept_ratio", type=float, default=0.5,
                        help='accept ratio of loss improve (default: 0.5)')
    parser.add_argument("--EPS", type=float, default=1e-8,
                        help='hyper parameter, close to zero')
    parser.add_argument("--use_clipped_value_loss", action='store_false', default=True,
                        help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='value loss clip parameter (default: 0.2)')
    parser.add_argument("--value_loss_coef", type=float, default=1,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument("--num_mini_batch", type=int, default=2,
                        help='number of batches for trpo (default: 1)')
    parser.add_argument("--use_max_grad_norm", action='store_false', default=True,
                        help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false', default=True,
                        help='use generalized advantage estimation')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--use_proper_time_limits", action='store_true', default=False,
                        help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True,
                        help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks", action='store_false', default=True,
                        help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks", action='store_false', default=True,
                        help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0,
                        help="coefficience of huber loss.")
    parser.add_argument("--line_search_fraction", type=float, default=0.5,
                        help="line search step size")
    parser.add_argument("--fraction_coef", type=float, default=0.05,
                        help="the coef of line search step size")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", type=bool, default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument("--save_interval", type=int, default=1,
                        help="time duration between contiunous twice models saving.")
    parser.add_argument("--log_interval", type=int, default=5,
                        help="time duration between contiunous twice log printing.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=True,
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False,
                        help="render or train.")
    parser.add_argument("--render_episodes", type=int, default=1,
                        help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1,
                        help="the play interval of each rendered image in saved video.")
    parser.add_argument("--model_dir", type=str, default="xxx",
                        help="by default None. set the path to pretrained model.")

    return parser
