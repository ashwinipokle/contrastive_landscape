import numpy as np
import argparse
from config import config
from config import update_config
from pathlib import Path
from data_model.gen_sparse_coding_data import *
import matplotlib.pyplot as plt 

"""
This is implementation of sparse coding algorithm in 
Simple, Efficient, and Neural Algorithms for Sparse Coding by Arora et. al.
"""
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
    parser.add_argument('--nn', 
                        help="number of training examples", # p
                        default=100,
                        required=False,
                        type=int)
    parser.add_argument('--NUM_EPOCHES', 
                        help="number of training steps",
                        default=100000,
                        required=False,
                        type=int)
    parser.add_argument('--c', 
                        help="parameter to control initialization of W",
                        default=1,
                        required=False,
                        type=float)
    parser.add_argument('--p', 
                        help="value of dimension p", 
                        default=20,
                        required=False,
                        type=int)
    parser.add_argument('--d', 
                        help="value of dimension d", 
                        default=20,
                        required=False,
                        type=int)
    parser.add_argument('--m', 
                        help="value of dimension m", 
                        default=20,
                        required=False,
                        type=int)
    parser.add_argument('--lr', 
                        help="learning rate",
                        default=5e-4,
                        required=False,
                        type=float)
    parser.add_argument('--sigma0', 
                        help="variance of gaussian noise being added to groundtruth",
                        default=None,
                        required=False,
                        type=float)  
    # Condition 2 in Model definition; Page 4 
    parser.add_argument('--threshold', 
                        help="value of thresholding constant",
                        default=0.7,
                        required=False,
                        type=float)                         
    args = parser.parse_args()
    update_config(config, args)
    return args

def verify_delta_k_close(A, Aopt, delta_factor=50, k=1):
    n = A.shape[1]
    delta = 1 / np.log(n)
    for i in range(n):
        diff_norm = np.linalg.norm(A[:, i] - Aopt[:, i], ord=2)
        assert diff_norm <= delta_factor * delta, f"diff_norm {diff_norm}, delta {delta_factor * delta}"

    opt_norm = np.linalg.norm(Aopt, ord=2)
    mat_diff_norm = np.linalg.norm(A - Aopt, ord=2)
    assert mat_diff_norm <= k*opt_norm, f"Opt norm {k * opt_norm} diff_norm {mat_diff_norm}"

# Used to check if projection onto set B is needed in Olshausen field update
def verify_delta_close(A, Aopt, delta_factor=50):
    n = A.shape[1]
    delta = 1 / np.log(n)

    for i in range(n):
        diff_norm = np.linalg.norm(A[:, i] - Aopt[:, i], ord=2)
        assert diff_norm <= delta_factor * delta, f"diff_norm {diff_norm}, delta {delta_factor * delta}"
    mat_norm = np.linalg.norm(A.T, ord=2)
    opt_norm = np.linalg.norm(Aopt, ord=2)
    assert mat_norm <= 2*opt_norm, f"Opt norm {2 * opt_norm} diff_norm {diff_norm}"

# Implementation of Algorithm 2 in Simple, Efficient, and Neural Algorithms for Sparse Coding by Arora et. al.
# Data is generated as x = Mz + e
# Algorithm assumes that e is negligible
# Given x_1, x_2, ..., x_p find basis vectors M_1, M_2, ..., M_k and 
# sparse vectors z_1, z_2, ..., z_p that minimize reconstruction error
def sparse_coding_neural_update(x, M, C, eta, num_steps, Mopt,zopt):
    # I assume that M is initialized appropriately (\delta, 2)
    # TODO: Implement initialization algorithm
    # NOte: ^ Probably not needed right now bc we are verifying our sparse model
    zdistlist = []
    distlist = []
    losslist = []
    slist = []
    for s in range(num_steps):
        # Decode step
        z = x @ M.T #n*m
        z_mask = np.maximum(np.abs(z), C/2) #
        z[z_mask <= C/2] = 0

        # Update step
        sign_z = np.ones_like(z)
        sign_z[z < 0] = -1
        sign_z[z == 0] = 0

        grad = (x - z @ M).T @ sign_z # p*n n*m = p*m 
        grad /= x.shape[0] # n

        M = M + eta * grad.T

        verify_delta_k_close(M, Mopt, k=2)

        loss = np.round(np.linalg.norm(x - z @ M, 'fro'),5)
        dist = np.round(np.linalg.norm(M.T - Mopt, 'fro'),5)
        zdist = np.round(np.linalg.norm(z.T - zopt, 'fro'),5)
        
        if not s % 50:
            print(f" Step {s}/{num_steps} Loss {loss} Dist {dist} Z {zdist}")      
            losslist.append(loss)  
            distlist.append(dist) 
            zdistlist.append(zdist)
            slist.append(s) 

    return M, z.T, losslist, distlist, zdistlist, slist

def olshausen_field_update(x, M, C, M_opt, eta, num_steps):
    for s in range(num_steps):
        # Decode step
        z = x @ M.T 
        z_mask = np.maximum(np.abs(z), C/2)
        z[z_mask <= C/2] = 0

        # Update step
        grad = (x - z @ M).T @ z
        grad /= x.shape[0]

        if not (num_steps % 20):
            print(f" Step {s}/{num_steps} Loss {np.linalg.norm(x - z @ M)}")

        M = M + eta * grad.T
        verify_delta_close(M, M_opt)
        # No need to project onto set B if M is delta close to Mopt for all steps
    return M, z.T

def main(args):
    M = gen_M(p=args.p, d=args.d) # p*d

    Z = gen_z(n=args.nn, d=args.d, prob=None) # d*n
    epsilon = gen_epsilon(n=args.nn, p=args.p, d=args.d, sigma0=args.sigma0)

    X = (M @ Z + epsilon).T  # n*p

    W_init = gen_Winit(M, p=args.p, m=args.m, d=args.d, c=0).T # m*p, m=d

    verify_delta_k_close(W_init, M, k=2)

    M_final, z_final, losslist, distlist, zdistlist, slist = sparse_coding_neural_update(X, W_init, args.threshold, args.lr, args.NUM_EPOCHES, M,Z)

    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(18,5), gridspec_kw={'width_ratios': [1, 1, 1]})
    ax1.plot(slist, losslist, color='k', linewidth=1.0) 
    ax1.set(xlabel="epochs",ylabel="loss")

    ax2.plot(slist,distlist, color='k', linewidth=1.0) 
    ax2.set(xlabel="epochs",ylabel="dictionary error")

    ax3.plot(slist,zdistlist, color='k', linewidth=1.0) 
    ax3.set(xlabel="epochs",ylabel="latents error")
    
    plt.savefig("{0}/result.png".format(args.logDir))
    plt.close()

if __name__ == '__main__':
    args = parse_args()

    print('config.n ', args.nn)  # debug
    print('config.p ', args.p)  # debug
    print('config.d ', args.d)  # debug
    print('config.m ', args.m)  # debug
    print('config.lr ', args.lr)  # debug
    print('config.c', args.c)

    # Save Result
    root_output_dir = Path(config.LOG_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    main(args)