import torch
from torch import nn
import math
import random
import numpy as np
import time

def relative_error(x, y, p='fro', dim=None, keepdim=False, out=None, dtype=None):
    error = torch.norm(x-y, p=p, dim=dim, keepdim=keepdim, out=out, dtype=dtype)
    denorm =  1 / torch.norm(x+y, p=p, dim=dim, keepdim=keepdim, out=out, dtype=dtype)
    return error * denorm

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

    q_approx = torch.cat(q_list, dim=-1) 
    k_approx = torch.cat(k_list, dim=-1) 
    return q_approx, k_approx

def softmax_approx_error(q, k, order_list=[]):

    attn_weights = torch.matmul(q, k.transpose(2, 3)) 
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    q_approx, k_approx = poly_approx(q, k, order_list=order_list)

    attn_weights_approx = torch.matmul(q_approx, k_approx.transpose(2, 3))
    
    # attn_norm_approx = torch.sum(attn_weights_approx, dim=-1, keepdim=True)

    b, h, n, r = q_approx.shape
    attn_norm_approx_2 = torch.matmul(k_approx.transpose(2, 3), torch.ones((b, h, n, 1), dtype=q_approx.dtype, device=q_approx.device))
    attn_norm_approx_2 = torch.matmul(q_approx, attn_norm_approx_2)

    # b, h, n, 1
    attn_norm_approx = attn_norm_approx_2


    attn_weights_approx = attn_weights_approx / attn_norm_approx
    return relative_error(attn_weights, attn_weights_approx, p='fro', dim=(-1,-2))

def get_total_order_dim(order_list=[]):
    dim = 1
    for i, r in enumerate(order_list):
        dim += r ** (i+1)
    return dim

def approx_attention_output(q, k, v, order_list=[]):
    # b, h, n, r where r greater than d, but small than n
    q_approx, k_approx = poly_approx(q, k, order_list=order_list)
    b, h, n, r = q_approx.shape
    attn_norm_approx = torch.matmul(k_approx.transpose(2, 3), torch.ones((b, h, n, 1), dtype=q_approx.dtype, device=q_approx.device))
    attn_norm_approx = torch.matmul(q_approx, attn_norm_approx)

    # # b, h, n, 1
    # attn_norm_approx = attn_norm_approx_2

    # b, h, n, r
    k_approx = k_approx / attn_norm_approx

    # b, h, r, d
    kv_approx = torch.matmul(k_approx.transpose(2, 3), v)

    # b, h, n, d
    approx_attn_output = torch.matmul(q_approx, kv_approx)

    return approx_attn_output

def actual_attention_output(q, k, v):
    attn_weights = torch.matmul(q, k.transpose(2, 3)) 

    # attn_weights_norm = torch.matmul(attn_weights, torch.ones())

    # attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    attn_output = torch.matmul(attn_weights, v)

    return attn_output

if __name__ == "__main__":
    b, h, n, d = 1, 1, 10000, 8
    n_list = [2**10, 2**11, 2**12, 2**13]
    # d_list = [2,4,6,8,10,12]
    d_list = [8,9,10,11,12]
    # d_list = [13]
    # d_list = [2,4,6,8]
    dtype = torch.float32
    seeds = [0,1,2]

    random.seed(seeds[0])
    np.random.seed(seeds[1])
    torch.manual_seed(seeds[2])

    for d in d_list:
        n = 10 * 2**d
        # q = torch.randn(b, h, n, d, dtype=dtype).cuda() / d ** 0.5
        # k = torch.randn(b, h, n, d, dtype=dtype).cuda() / d ** 0.5
        # v = torch.randn(b, h, n, d, dtype=dtype).cuda()
        q = torch.randn(b, h, n, d, dtype=dtype) / d ** 0.5
        k = torch.randn(b, h, n, d, dtype=dtype) / d ** 0.5
        v = torch.randn(b, h, n, d, dtype=dtype) 


        error_0 = softmax_approx_error(q, k, order_list=[d,d])


        time1 = time.time()

        actual_attn = actual_attention_output(q, k, v)


        time2 = time.time()

        approx_attn = approx_attention_output(q, k, v, order_list=[d,d])

        time3 = time.time()

        error_1 = relative_error(actual_attn, approx_attn, p='fro', dim=(-1,-2))


        # for i, r in enumerate(n_list):

        actual_time = time2 - time1
        approx_time = time3 - time2
        print(n, error_0, error_1, actual_time, approx_time)
