import os
import argparse
from solver_encoder import Solver
from data_loader import get_loader
from torch.backends import cudnn
import json


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Data loader.
    vcc_loader = get_loader(config.data_dir, config.batch_size, config.len_crop)

    save_path = os.path.join(config.save_dir, config.model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(vars(config), f, indent=4)

    config.save_dir = save_path
    solver = Solver(vcc_loader, config)

    solver.train()
        
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=32)

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for Adam optimizer')
    
    # Training configuration.
    parser.add_argument('--data_dir', type=str, default='./spmel')
    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    
    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=100)

    parser.add_argument('--save_step', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--model_name', type=str, default='autovc')

    parser.add_argument('--initial_step', type=int, default=0, help='initial step for training')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='path to load checkpoints')
    
    config = parser.parse_args()
    print(config)
    
        
    main(config)