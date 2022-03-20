import argparse
from config import config
from config import update_config

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='Train dual network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--model',
                        help='Choose one among simplified, simsiam and simclr',
                        type=str,
                        default="simplified")
    parser.add_argument('--local_rank',
                        type=int,
                        default=0)
    parser.add_argument('--normalize_repr',
                        help='Whether to normalize representations while training',
                        type=str2bool,
                        default=False)
    parser.add_argument('--ema_decay',
                        help='The decay parameter for exponential moving average update',
                        type=float,
                        default=None)
    parser.add_argument('--use_masking',
                        help='Whether to use masked data (independent masking)',
                        type=str2bool,
                        default=False)
    parser.add_argument('--use_multimasking',
                        help='Whether to use masked data loader for randomized masking of inputs',
                        type=str2bool,
                        default=False)
    parser.add_argument('--n_aug',
                        help='Number of augmentations per input if using multimasking',
                        type=int,
                        default=5)
    parser.add_argument('--use_bn',
                        help='Whether to use batch norm in simplified model',
                        type=str2bool,
                        default=False)
    parser.add_argument('--lr',
                        help='learning rate',
                        type=float,
                        default=0.025)
    parser.add_argument('--batch_size',
                        help='training batch size',
                        type=int,
                        default=384)
    parser.add_argument('--temperature',
                        help='temperature in simclr loss',
                        type=float,
                        default=0.05)
    parser.add_argument('--use_pred',
                        help='Include a predictor in the model?',
                        type=str2bool,
                        default=False)
    parser.add_argument('--use_bias',
                        help='should the encoder have a bias',
                        type=str2bool,
                        default=True)
    parser.add_argument('--use_pred_bias',
                        help='Should the predictor have a bias?',
                        type=str2bool,
                        default=True)
    parser.add_argument('--m_identity',
                        help='Use identity for M',
                        type=str2bool,
                        default=False) 
    parser.add_argument('--use_alt_norm',
                    help='Use alternate normalization (i.e. column normalization) of weight matrices Wo and Wt (supported in simplified and simplified-no-alter).',
                    type=str2bool,
                    default=False)
    parser.add_argument('--use_row_norm',
                    help='Row normalize the weight matrices Wo and Wt (supported in simplified and simplified-no-alter).',
                    type=str2bool,
                    default=False)
    parser.add_argument('--log_metrics',
                    help="should we log wandb metrics?",
                    action='store_true'
                    )
    parser.add_argument('--const_bias',
                    help="should the bias of encoder be a constant",
                    action='store_true'
                    )
    parser.add_argument('--bias_val',
                    help="The value of biases of encoder, if constant",
                    type=float,
                    default=0.005
                    )
    parser.add_argument('--wandb_project',
                    help="Name of project for purposes of wandb logging",
                    type=str,
                    default="experiment-log"
                    )
    args = parser.parse_args()
    update_config(config, args)
    return args
