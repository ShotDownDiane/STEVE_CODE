import numpy as np
import math

import torch 
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Module
from torch import tensor
import torch.nn.init as init
import torch.nn.functional as F


## gradient reverse layer
class RevGradFunc(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None

revgrad = RevGradFunc.apply

class RevGradLayer(Module):
    def __init__(self, alpha=0.01):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__()

        self._alpha = tensor(alpha, requires_grad=False)

    def forward(self, input_ , p):
        alpha=self._alpha/np.power((1+10*p),0.75)# todo
        return revgrad(input_, alpha)


def cal_cheb_polynomial(laplacian, K):
    """
    Compute the Chebyshev Polynomial, according to the graph laplacian.

    :param laplacian: the graph laplacian, [N, N].
    :return: the multi order Chebyshev laplacian, [K, N, N].
    """
    N = laplacian.size(0)  # [N, N]
    multi_order_laplacian = torch.zeros([K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
    multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

    if K == 1:
        return multi_order_laplacian
    else:
        multi_order_laplacian[1] = laplacian
        if K == 2:
            return multi_order_laplacian
        else:
            for k in range(2, K):
                multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - \
                                           multi_order_laplacian[k - 2]

    return multi_order_laplacian

def cal_laplacian(graph):
    """
    return the laplacian of the graph.

    :param graph: the graph structure **without** self loop, [N, N].
    :param normalize: whether to used the normalized laplacian.
    :return: graph laplacian.
    """
    I = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
    graph = graph + I  # add self-loop to prevent zero in D
    D = torch.diag(torch.sum(graph, dim=-1) ** (-0.5))
    L = I - torch.mm(torch.mm(D, graph), D)
    return L


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1)

    def forward(self, x):  # x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  # return: (batch_size, c_out, input_length-1+1, num_nodes-1+1)


class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        """
        :param x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        :return: (batch_size, c_out, input_length-kt+1, num_nodes)
        """
        x_in = self.align(x)[:, :, self.kt - 1:, :]  # (batch_size, c_out, input_length-kt+1, num_nodes)
        if self.act == "GLU":
            # x: (batch_size, c_in, input_length, num_nodes)
            x_conv = self.conv(x)
            # x_conv: (batch_size, c_out * 2, input_length-kt+1, num_nodes)  [P Q]
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
            # return P * sigmoid(Q) shape: (batch_size, c_out, input_length-kt+1, num_nodes)
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)  # residual connection
        return torch.relu(self.conv(x) + x_in)  # residual connection


class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out, device):
        super(SpatioConvLayer, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks).to(device))  # kernel: C_in*C_out*ks
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1).to(device))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x,Lk):
        # Lk: (Ks, num_nodes, num_nodes)
        # x:  (batch_size, c_in, input_length, num_nodes)
        # x_c: (batch_size, c_in, input_length, Ks, num_nodes)
        # theta: (c_in, c_out, Ks)
        # x_gc: (batch_size, c_out, input_length, num_nodes)
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)  # delete num_nodes(n)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b  # delete Ks(k) c_in(i)
        x_in = self.align(x)  # (batch_size, c_out, input_length, num_nodes)
        return torch.relu(x_gc + x_in)  # residual connection


class STConvBlock(nn.Module):
    def __init__(self, ks, kt, n, c, p, device):
        super(STConvBlock, self).__init__()
        self.tconv1 = TemporalConvLayer(kt, c[0], c[1], "GLU")
        self.sconv = SpatioConvLayer(ks, c[1], c[1],device)
        self.tconv2 = TemporalConvLayer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x,graph):  # x: (batch_size, feature_dim/c[0], input_length, num_nodes)
        x_t1 = self.tconv1(x)    # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_s = self.sconv(x_t1,graph)   # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_t2 = self.tconv2(x_s)  # (batch_size, c[2], input_length-kt+1-kt+1, num_nodes)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)
    

class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(1)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        if attn_mask is not None:

            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


def mean_subtraction(train_data, test_data):
    """ Zero mean samples """
    train_data_mean = np.mean(train_data, axis=0)

    train_data -= train_data_mean
    if test_data:
        test_data -= train_data_mean
        samples = train_data.tolist()+test_data.tolist()
    else:
        samples = train_data
    return samples


def pca(train_data, test_data=[], min_pov=0.90):
    """ Apply PCA to dataset """
    def propose_suitable_d(eigenvalues):
        """ Propose a suitable d using POV = 95% """
        sum_D = sum(eigenvalues)
        for d in range(0, len(eigenvalues)):
            pov = sum(eigenvalues[:d])/sum_D
            if pov > min_pov:
                return d

    # Zero-mean train data and test data
    samples = mean_subtraction(train_data, test_data)
    samples = np.asarray(samples)
    cov = np.dot(samples.T, samples) / samples.shape[0]

    eigenvectors, eigenvalues, _ = np.linalg.svd(cov)

    samples = np.dot(samples, eigenvectors)  # decorrelate the data

    d = propose_suitable_d(eigenvalues)  # find suitable d

    # samples_pca becomes [N x d]
    samples_pca = np.dot(samples, eigenvectors[:, :d])
    return samples_pca, samples, eigenvalues


def pca_whitening(train_data, test_data=[], min_pov=0.90):
    """ 
        whiten the data,
        The whitened data will be a gaussian with zero mean and identity covariance matrix.
    """
    _, samples, eigenvalues = pca(train_data, test_data, min_pov=min_pov)
    # whiten the data:
    # divide by the eigenvalues (which are square roots of the singular values)
    samples_pca_white = samples / np.sqrt(eigenvalues + 1e-5)
    return samples_pca_white




class MLPAttention(nn.Module):
    def __init__(self, d_models):
        super().__init__()
        self.d_models=d_models
        self.mlp=nn.Sequential(
            nn.Linear(2*d_models,d_models),
            nn.ReLU(),
            nn.Linear(d_models,1),
        )
        self.softmax = nn.Softmax(-1)
        self.tau=4 # 1,2,4
    
    def forward(self, Q, K, V, attn_mask=None):
        # q=bnd,k=bkd,v=bkd
        
        k=K.shape[1]
        n=Q.shape[1]
        b=Q.shape[0]
        
        Q=Q.unsqueeze(2)# bn1d
        Q=Q.repeat(1,1,k,1)#bnkd[]
        K=K.unsqueeze(1)#b1kd
        K=K.repeat(1,n,1,1)#bnkd
        input=torch.stack([Q,K],-1).reshape(b,n,k,-1)# bnk 2d
        res=self.mlp(input).squeeze(-1)#bnk
        # weather
        # res[20,:,59]=torch.tensor(-99)
        # res[20,:,18]=torch.tensor(99)
        # import pdb
        # pdb.set_trace()

        att = self.softmax(res/self.tau) # 大gamma

        out=torch.bmm(att,V) # bnd

        return out,att



class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
