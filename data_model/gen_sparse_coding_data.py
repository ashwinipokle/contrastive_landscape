import numpy as np
import pickle

def gen_z_random(n, d, prob=None):
    if prob is None:
        prob = np.log(np.log(d))/d
    else:
        prob = (np.log(np.log(d))/d)**prob  
    return np.random.choice([0,-1,1], (d, n), p=[1-prob, prob/2, prob/2]) 

def gen_z(n, d, prob=None, one_hot_latent=False):
    z = np.zeros(d)
    if one_hot_latent:
        (num_neg_ones, num_ones) = np.random.permutation([0, 1])
        s1 = num_neg_ones
        s2 = s1 + num_ones
    else:
        if prob is None:
            prob = np.log(np.log(d))/d
        else:
            prob = prob
        s1 = np.ceil(d*prob/2)
        s2 = max(np.ceil(d*prob),np.ceil(d*prob/2)+1)
    z[:int(s1)] = -1
    z[int(s1):int(s2)] = 1
    print(f"Sparsity {len(np.where(z == 0)[0])} Total entries {z.shape[0]}")
    return np.array([np.random.permutation(z) for i in range(n)]).T

def gen_z_one_hot(n, d, prob=None):
    z = np.zeros((n, d))
    ridx = np.arange(n)
    cidx = np.random.choice(np.arange(d), n)
    z[ridx, cidx] = 1
    print(f"Sparsity {len(np.where(z == 0)[0])} Total entries {z.shape[0]}")
    return z.T

# ensure that atleast 1 entry is non zero in every column
def gen_z_k_sparse(n, d, prob=None):
    z = np.random.choice([0,1], (d, n), p=[1-prob, prob]) 
    for c in range(d):
        if sum(z[c]) == 0:
            ridx = np.arange(n)
            z[c, ridx] = 1
    print(f"Sparsity {len(np.where(z == 0)[0])} Total entries {z.shape[0]}")
    return z

def gen_M(p, d):
    """
    Generate a column orthonormal matrix of shape (p, d)
    """
    X = np.random.normal(0, 1, (p, d))
    assert p >= d
    Q, _ = np.linalg.qr(X)
    return Q

# Xavier normal
def random_normal_weight_init(input, output):
    std = np.sqrt(2)/np.sqrt(input + output)
    return np.random.normal(0,std,(input,output))

# Xavier uniform
def random_weight_init(input,output):
    b = np.sqrt(6)/np.sqrt(input + output)
    return np.random.uniform(-b,b,(input, output))

def kaiming_weight_init(input, output, fanmode='fan_in'):
    dim = output
    if fanmode == 'fan_in':
        dim = input
    b = np.sqrt(3)/np.sqrt(dim)
    return np.random.uniform(-b,b,(input, output))

def gen_Winit(M, p, m, d, c=None):
    """
    Generate initialization of W0 based on M
    """
    if m>d:
        MS = M[:,np.random.choice(d,m)]
    else:
        MS = M
    if c is None: # random
        return np.random.normal(0, np.sqrt(1/(p*d)), (p, m))
    elif c > 0: # close to M
        return MS + np.random.normal(0, c, (p, m)) #np.random.normal(0, 1.0/p**(c/2), (p, m)) 
    else: # equals M
        return MS    

def gen_epsilon(n, p, d, sigma0=None):
    if sigma0 is None:
        sigma0 = np.sqrt(np.log(d))/d # default, following yuanzhi
        return np.random.normal(0, sigma0, (p, n))
    elif sigma0==0:
        return np.zeros((p, n))
    else:    
        sigma0 = (np.sqrt(np.log(d))/d)**sigma0 # smaller
        return np.random.normal(0, sigma0, (p, n))



