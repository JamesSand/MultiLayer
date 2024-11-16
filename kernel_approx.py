import torch
from torch import nn
import math
import random
import numpy as np

def relative_error(x, y, p='fro', dim=None, keepdim=False, out=None, dtype=None):
    error = torch.norm(x-y, p=p, dim=dim, keepdim=keepdim, out=out, dtype=dtype)
    denorm =  1 / torch.norm(x+y, p=p, dim=dim, keepdim=keepdim, out=out, dtype=dtype)
    return error * denorm


# Q K.T = U_Q @ S_Q @ V_Q.T @ (U_K @ S_K @ V_K.T).T, Q, K \in R^{n \times d}
# Q K.T = U_Q @ S_Q @ V_Q.T @ V_K @ (U_K @ S_K).T
# get SVD([K, v]) easily from SVD(K); K \in \R^{n \times d} o(n) time
# incremental SVD / online SVD
def sketch_reform(key_states):
    # run approx svd solver for tall matrix
    # see detail in https://pytorch.org/docs/stable/generated/torch.linalg.svd.html#torch.linalg.svd
    input_dtype = key_states.dtype
    b, h, n, d = key_states.shape
    if n > d: 
        full_matrices = False
    else:
        full_matrices = True
    if input_dtype == torch.float16:
        key_states = key_states.to(torch.float32)
    kU, kS, kVh = torch.linalg.svd(key_states, full_matrices=full_matrices, driver='gesvda') # solver: None, gesvd, gesvdj, gesvda
    if torch.max(torch.abs(kU)) + torch.max(torch.abs(kVh)) > 60000: # if fail roll back to general svd solver
        kU, kS, kVh = torch.linalg.svd(key_states, full_matrices=full_matrices, driver='gesvd')
    
    sketch = kVh.transpose(2, 3)
    if input_dtype == torch.float16:
        sketch = sketch.to(torch.float16)
        kS = kS.to(torch.float16)
    return sketch, kS

CHAR_LIST = 'abcdefghijklmnopqrstuvw'
# TODO: accelerate below
def order_approx(x, order=1):
    assert order <= 23
    shape = list(x.shape)
    shape[-1] = -1
    denorm = (1 / math.factorial(order)) ** 0.5

    operation_in = [f'xyz{CHAR_LIST[i]}' for i in range(order)]
    operation_in = ','.join(operation_in)
    operation_out = [f'{CHAR_LIST[i]}' for i in range(order)]
    operation_out = 'xyz' + ''.join(operation_out)
    operation = operation_in+'->'+operation_out # xyza,xyzb,xyzc,xyzd,xyze->xyzabcde
    x_command = ', '.join(['x'] * order)
    # torch.einsum('xyza,xyzb,xyzc,xyzd,xyze->xyzabcde', x, x, x, x, x).view(shape)
    torch_command = f"torch.einsum('{operation}', {x_command}).view(shape)" 

    # this is really fancy
    x_order = eval(torch_command)

    # breakpoint()
    # print()

    return x_order * denorm # (q.T k)^d / d!

# len(order_list) is the highest order we are going to approximate
# the first order_list[i] coordinate doing i-th order approximate
def poly_approx(q, k, order_list=[]):
    # print(get_total_order_dim(order_list))
    device = q.get_device()
    shape = list(q.shape)
    dtype = q.dtype
    shape[-1] = 1
    # breakpoint()
    if device >=0:
        q_list = [torch.ones(shape, dtype=dtype).to(device=f'cuda:{device}')]
        k_list = [torch.ones(shape, dtype=dtype).to(device=f'cuda:{device}')]
    else:
        q_list = [torch.ones(shape, dtype=dtype).cpu()]
        k_list = [torch.ones(shape, dtype=dtype).cpu()]

    for i, r in enumerate(order_list):
        if r > 1:
            q_list.append(order_approx(q[:,:,:,:r], order=i+1))
            k_list.append(order_approx(k[:,:,:,:r], order=i+1))
        else:
            assert r == 1
            denorm = (1 / math.factorial(i+1)) ** 0.5
            q_list.append((q[:,:,:,:r] ** (i+1)) * denorm)
            k_list.append((k[:,:,:,:r] ** (i+1)) * denorm)
    
    # for q_item in q_list:
    #     print(q_item.shape)

    # breakpoint()
    # print()

    q_approx = torch.cat(q_list, dim=-1) 
    k_approx = torch.cat(k_list, dim=-1) 

    # breakpoint()
    # print()

    return q_approx, k_approx

def exp_approx_error(q, k, order_list=[]):
    attn_weights = torch.matmul(q, k.transpose(2, 3)) 
    attn_weights = torch.exp(attn_weights)

    q_approx, k_approx = poly_approx(q, k, order_list=order_list)
    attn_weights_approx = torch.matmul(q_approx, k_approx.transpose(2, 3))

    # breakpoint()
    # print()

    return relative_error(attn_weights, attn_weights_approx, p='fro', dim=(-1,-2))

def softmax_approx_error(q, k, order_list=[]):
    attn_weights = torch.matmul(q, k.transpose(2, 3)) 
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    q_approx, k_approx = poly_approx(q, k, order_list=order_list)

    # print(order_list)
    # print(q.shape)
    # print(k.shape)
    # print(q_approx.shape)
    # print(k_approx.shape)

    # breakpoint()

    attn_weights_approx = torch.matmul(q_approx, k_approx.transpose(2, 3))
    
    attn_norm_approx = torch.sum(attn_weights_approx, dim=-1, keepdim=True)
    attn_weights_approx = attn_weights_approx / attn_norm_approx
    return relative_error(attn_weights, attn_weights_approx, p='fro', dim=(-1,-2))

def get_total_order_dim(order_list=[]):
    dim = 1
    for i, r in enumerate(order_list):
        dim += r ** (i+1)
    return dim

if __name__ == "__main__":
    b, h, n, d = 1, 1, 10000, 8
    n_list = [2**10, 2**11, 2**12, 2**13]
    d_list = [2,4,6,8,10,12]
    dtype = torch.float32
    seeds = [0,1,2]

    random.seed(seeds[0])
    np.random.seed(seeds[1])
    torch.manual_seed(seeds[2])

    for d in d_list:
        n = 10 * 2**d
        # q = torch.randn(b, h, n, d, dtype=dtype).cuda() / d ** 0.5
        # k = torch.randn(b, h, n, d, dtype=dtype).cuda() / d ** 0.5
        q = torch.randn(b, h, n, d, dtype=dtype) / d ** 0.5
        k = torch.randn(b, h, n, d, dtype=dtype) / d ** 0.5
        error_0 = softmax_approx_error(q, k, order_list=[d,d])
        print(n, error_0)
    # error_1 = softmax_approx_error(q, k, order_list=[32])
    # error_2 = softmax_approx_error(q, k, order_list=[32,12,5])
    # error_3 = softmax_approx_error(q, k, order_list=[32,32,12,4,3,2,2,2,2,2])
    # print(error_0)
    # print(error_1)
    # print(error_2)
    # print(error_3)
    # print()
    # print(error_0 - error_1)
    # print(error_1 - error_2)
    # print(error_2 - error_3)
    # query_states, key_states, sketch = sketch_reform(q, k)
    # print(relative_error(k @ kVh.transpose(2, 3), key_states))
    # print(sketch.shape)
    # print(q.shape)
    # print(query_states.shape)
