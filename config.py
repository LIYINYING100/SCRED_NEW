import argparse

arg_lists = []

parser = argparse.ArgumentParser(description='dml')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--data_path', type=str, default='./AAPM-Mayo-CT-Challenge/')
data_arg.add_argument('--saved_path', type=str,default='') 
data_arg.add_argument('--saved_path1', type=str,default='') 
data_arg.add_argument('--transform', type=bool, default=False)
data_arg.add_argument('--patch_n', type=int, default=9,
                      help='if patch training, batch size is (--patch_n * --batch_size')
data_arg.add_argument('--patch_size', type=int, default=64)  # 64*64
data_arg.add_argument('--batch_size', type=int, default=1)
data_arg.add_argument('--shuffle', type=bool, default=True)
data_arg.add_argument('--num_workers', type=int, default=7,
                      help='# of subprocesses to use for data loading')  # 7
data_arg.add_argument('--load_mode', type=int, default=0,
                      help='If the available memory(RAM) is more than 10GB, it is faster to run --load_mode=1')

data_arg.add_argument('--norm_range_min', type=float, default=-1024)  # -1024.0)-0.1
data_arg.add_argument('--norm_range_max', type=float, default=3072)  # 3072.0)0.3
data_arg.add_argument('--trunc_min', type=float, default=-160)  # -160.0)-0.01
data_arg.add_argument('--trunc_max', type=float, default=240)  # 240.0)240


# Training params

train_arg = add_argument_group('Training Params')
train_arg.add_argument('--mode', type=str, default='train')
train_arg.add_argument('--save_path', type=str, default='./2')
train_arg.add_argument('--device', type=str, default='cuda')
train_arg.add_argument('--multi_gpu', type=bool, default=False)

train_arg.add_argument('--model_num', type=int, default=2,
                    help='Number of models to train for DML')
train_arg.add_argument('--num_epochs', type=int, default=100)
train_arg.add_argument('--lr', type=float, default=1e-5)

train_arg.add_argument('--decay_iters', type=int, default=3000)
train_arg.add_argument('--save_iters', type=int, default=10000)


parser.add_argument('--test_patient', type=str, default='L506')  # 'L506') P107
parser.add_argument('--test_iters', type=int, default=210000)
parser.add_argument('--result_fig', type=bool, default=True)
parser.add_argument('--print_iters', type=int, default=20)
parser.add_argument('--T', type=int, default=20)
parser.add_argument('--alpha', type=int, default=0.1)

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--best', type=bool, default=False,
                      help='Load best model or most recent for testing')
# misc_arg.add_argument('--random_seed', type=int, default=1,
#                       help='Seed to ensure reproducibility')
misc_arg.add_argument('--logs_dir', type=str, default='logs',
                      help='Directory in which Tensorboard logs wil be stored')
misc_arg.add_argument('--use_tensorboard', type=bool, default=True,
                      help='Whether to use tensorboard for visualization')
misc_arg.add_argument('--resume', type=bool, default=False,
                      help='Whether to resume training from checkpoint') 
misc_arg.add_argument('--ckpt_path', type=str, default='ckpt',
                      help='Directory in which to save model checkpoints')
def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
