import torch
import torch.nn as nn
import torch.nn.functional as F
import spikingjelly.cext.functional
import warnings
import math
import numpy as np
import time
class SparseLinear(nn.Linear):
    '''
    * :ref:`API in English <SparseLinear-en>`
    .. _SparseLinear-en:

    :param in_features: size of each input sample
    :type in_features: int
    :param out_features: size of each output sample
    :type out_features: int
    :param bias: If set to ``False``, the layer will not learn an additive bias.
        Default: ``True``
    :type bias: bool

    The fully connected layer for sparse inputs. This module has a similar behavior as ``torch.nn.Linear``.

    .. admonition:: Warning
        :class: warning

        This function is implemented by converting ``sparse`` to a sparse format and doing a sparse matrix multiplication.
        If the sparsity of ``sparse`` is not high enough, the speed of this function will be slower than ``torch.mm``.

    .. admonition:: Warning
        :class: warning

        There are some numeral errors when doing the sparse matrix multiplication. But the errors are not significant.

    .. admonition:: Warning
        :class: warning

        This layer does not support to run on cpu.
    '''
    def forward(self, sparse: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            return spikingjelly.cext.functional.sparse_mm_dense(sparse, self.weight.t())
        else:
            return spikingjelly.cext.functional.sparse_mm_dense(sparse, self.weight.t()) + self.bias

class AutoSparseLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, in_spikes: bool = False):
        '''
        * :ref:`API in English <AutoSparseLinear-en>`

        :param in_features: size of each input sample
        :type in_features: int
        :param out_features: size of each output sample
        :type out_features: int
        :param bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        :type bias: bool
        :param in_spikes: Whether inputs are spikes, whose elements are 0 and 1
            Default: ``False``
        :type in_spikes: bool

        The auto sparse fully connected layer. For an input, if the corresponding critical sparsity of the input's batch
        size is unknown, this layer will firstly run the benchmark :ref:`AutoSparseLinear.benchmark <AutoSparseLinear.benchmark-en>` to get the critical sparsity. The critical sparsity is the sparsity where the sparse matrix multiplication and the dense matrix multiplication have the same speed. For an input, if the corresponding critical sparsity of the input's batch size is known, this layer can auto select whether using the sparse or dense matrix multiplication according to the current input's sparsity.

        .. admonition:: Warning
            :class: warning

            There are some numeral errors when doing the sparse matrix multiplication. But the errors are not significant.

        .. admonition:: Warning
            :class: warning

            This sparse matrix multiplication does not support to run on cpu. When this layer is on CPU, the dense matrix multiplication will be always used.

        '''
        super().__init__(in_features, out_features, bias)
        self.critical_sparsity = {}  
        self.in_spikes = in_spikes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.get_device() < 0:
            return F.linear(x, self.weight, self.bias)

        batch_size = x.shape[0]
        if batch_size not in self.critical_sparsity:
            self.benchmark(batch_size, x.device)

        csp = self.critical_sparsity[batch_size]
        if csp is None:
            return F.linear(x, self.weight, self.bias)

        else:
            with torch.no_grad():
                if self.in_spikes:
                    sparsity = 1 - x.mean().item()
                else:
                    sparsity = (x == 0).float().mean().item()
        if sparsity < csp:
            return F.linear(x, self.weight, self.bias)
        else:
            if self.bias is None:
                return spikingjelly.cext.functional.sparse_mm_dense(x, self.weight)
            else:
                return spikingjelly.cext.functional.sparse_mm_dense(x, self.weight) + self.bias    

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, critical_sparsity={self.critical_sparsity}'

    @torch.enable_grad()
    def benchmark(self, batch_size: int, device=None, run_times=1024, precision=1e-4, verbose=True):
        '''
        .. _AutoSparseLinear.benchmark-en:

        :param batch_size: batch size of the input
        :type batch_size: int
        :param device: where to running the benchmark. If ``None``, it will be set as same with this layer's device
        :type device: str
        :param run_times: the number of replicated running times for sparse/dense matrix multiplication. The benchmark
            result will be more reliable with a larger ``run_times``
        :type run_times: int
        :param precision: the precision of binary searching critical sparsity
        :type precision: float
        :param verbose: If ``True``, this function will print logs during running
        :type verbose: bool

        Using the binary search to find the critical sparsity when the batch size of the input is ``batch_size``. This function
        will run ``run_times`` sparse/dense matrix multiplication on different sparsity and compare their speeds until it
        finds the cirtical sparsity. If the dense matrix multiplication is faster than the sparse matrix multiplication
        when searching exceeds ``precision``, then the critical sparsity will be set to ``None``.

        '''
        if self.critical_sparsity.__len__() > 4:
            warnings.warn('AutoSparseLinear: The batch size of the input has changed more than 4 times. AutoSparseLinear may waste too much time on running benchmark.')

        if device is None:
            device = self.weight.device

        if verbose:
            print(f'{self} is running benchmark for batch_size={batch_size} at precision={precision} on device={device}')

        if self.bias is None:
            bias = False
        else:
            bias = True

        fc_sparse = SparseLinear(self.in_features, self.out_features, bias)
        fc_sparse.to(device)
        fc_dense = nn.Linear(self.in_features, self.out_features, bias)
        fc_dense.to(device)

        sparisity_r = 1.0
        sparisity_l = 0.1

        while True:
            sparisity = (sparisity_l + sparisity_r) / 2
            x = torch.rand(size=[batch_size, self.in_features], device=device)
            sparse = (x > sparisity).to(x)
            sparisity_a = (sparse == 0).to(x).mean().item()  
            t_list = []
            for _ in range(run_times * 2):
                fc_sparse.zero_grad()
                torch.cuda.synchronize()
                t_start = time.perf_counter()
                fc_sparse(sparse).sum().backward()
                torch.cuda.synchronize()
                t_list.append(time.perf_counter() - t_start)
            t_list = np.asarray(t_list)
            t_sparse = t_list[run_times:].sum()

            t_list = []
            for _ in range(run_times * 2):
                fc_dense.zero_grad()
                torch.cuda.synchronize()
                t_start = time.perf_counter()
                fc_dense(sparse).sum().backward()
                torch.cuda.synchronize()
                t_list.append(time.perf_counter() - t_start)
            t_list = np.asarray(t_list)
            t_dense = t_list[run_times:].sum()
            if verbose:
                print(f'sparisity_a={sparisity_a}, t_sparse={t_sparse}, t_dense={t_dense}')

            if t_sparse > t_dense:
                sparisity_l = sparisity_a
            elif t_sparse < t_dense:
                sparisity_r = sparisity_a
            else:
                break

            if sparisity_r - sparisity_l < precision:
                break

        if t_sparse < t_dense:
            self.critical_sparsity[batch_size] = sparisity_a
        else:
            self.critical_sparsity[batch_size] = None
        print(f'critical_sparsity[{batch_size}]={self.critical_sparsity[batch_size]}')
        del x, sparse, fc_sparse, fc_dense
        torch.cuda.empty_cache()



                
                




                


