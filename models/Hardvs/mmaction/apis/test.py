# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
# TODO import test functions from mmcv and delete them from mmaction2
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
import pdb
import numpy as np
import pandas as pd
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
try:
    from mmcv.engine import (collect_results_cpu, collect_results_gpu,
                             multi_gpu_test, single_gpu_test)
    from_mmcv = True
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'DeprecationWarning: single_gpu_test, multi_gpu_test, '
        'collect_results_cpu, collect_results_gpu from mmaction2 will be '
        'deprecated. Please install mmcv through master branch.')
    from_mmcv = False

if not from_mmcv:

    def single_gpu_test(model, data_loader):  # noqa: F811
        """Test model with a single gpu.

        This method tests model with a single gpu and
        displays test progress bar.

        Args:
            model (nn.Module): Model to be tested.
            data_loader (nn.Dataloader): Pytorch data loader.

        Returns:
            list: The prediction results.
        """
        print(f'************** TEST ***************')
        model.eval()
        True_label_class =  np.array(list(range(300)))
        # True_label = np.array(torch.tensor(True_label,dtype=torch.float32).unsqueeze(1))
        True_label=[]
        T_predict1=[]
        model.eval()
        results = []
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        for data in data_loader:
            with torch.no_grad():
                result = model(return_loss=False, **data)
            results.extend(result)

            # #################################################################
            # true_label = (data['label']).item()
            # topkresult = torch.tensor(result)
            # attn_topk, index = torch.topk(topkresult, k=1, dim=-1)
            # T_predict1.append(index.item())
            # True_label.append(true_label)
            # ##################################################################
            # use the first key as main key to calculate the batch size
            batch_size = len(next(iter(data.values())))
            for _ in range(batch_size):
                prog_bar.update()

        # pdb.set_trace() 
        # C= confusion_matrix(True_label, T_predict1,labels=True_label_class)
        # # plt.matshow(C, cmap=plt.cm.Greens) 
        # # plt.colorbar()
        # # for i in range(len(C)): 
        # #     for j in range(len(C)):
        # #         plt.annotate(C[i,j], xy=(i, j), horizontalalignment='center', verticalalignment='center')
        # # plt.ylabel('True label')
        # # plt.xlabel('Predicted label') 
        # # plt.imsave( '/DATA/wuzongzhen/wzz/hardvs_time_space/confusion_matrix.jpg', C) 


        # trans_mat = C
        # trans_prob_mat = (trans_mat.T/np.sum(trans_mat, 1)).T


        # label = ["Patt {}".format(i) for i in range(1, trans_mat.shape[0]+1)]
        # df = pd.DataFrame(trans_prob_mat, index=label, columns=label)

        
        # # Plot
        # plt.figure(figsize=(7.5, 6.3))
        # ax = sns.heatmap(df, xticklabels=df.corr().columns, 
        #                  yticklabels=df.corr().columns, cmap='magma',
        #                  linewidths=6, annot=True)
        
        # # Decorations
        # plt.xticks(fontsize=16,family='Times New Roman')
        # plt.yticks(fontsize=16,family='Times New Roman')
        
        # plt.tight_layout()
        # plt.savefig('/DATA/wuzongzhen/wzz/hardvs_time_space/confusion_matrix.jpg', transparent=True, dpi=800)

        # pdb.set_trace()

        return results

    def multi_gpu_test(  # noqa: F811
            model, data_loader, tmpdir=None, gpu_collect=True):
        """Test model with multiple gpus.

        This method tests model with multiple gpus and collects the results
        under two different modes: gpu and cpu modes. By setting
        'gpu_collect=True' it encodes results to gpu tensors and use gpu
        communication for results collection. On cpu mode it saves the results
        on different gpus to 'tmpdir' and collects them by the rank 0 worker.

        Args:
            model (nn.Module): Model to be tested.
            data_loader (nn.Dataloader): Pytorch data loader.
            tmpdir (str): Path of directory to save the temporary results from
                different gpus under cpu mode. Default: None
            gpu_collect (bool): Option to use either gpu or cpu to collect
                results. Default: True

        Returns:
            list: The prediction results.
        """
        model.eval()
        results = []
        dataset = data_loader.dataset
        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))
        for data in data_loader:
            with torch.no_grad():
                result = model(return_loss=False, **data)
            results.extend(result)

            if rank == 0:
                # use the first key as main key to calculate the batch size
                batch_size = len(next(iter(data.values())))
                for _ in range(batch_size * world_size):
                    prog_bar.update()

        # collect results from all ranks
        if gpu_collect:
            results = collect_results_gpu(results, len(dataset))
        else:
            results = collect_results_cpu(results, len(dataset), tmpdir)
        return results

    def collect_results_cpu(result_part, size, tmpdir=None):  # noqa: F811
        """Collect results in cpu mode.

        It saves the results on different gpus to 'tmpdir' and collects
        them by the rank 0 worker.

        Args:
            result_part (list): Results to be collected
            size (int): Result size.
            tmpdir (str): Path of directory to save the temporary results from
                different gpus under cpu mode. Default: None

        Returns:
            list: Ordered results.
        """
        rank, world_size = get_dist_info()
        # create a tmp dir if it is not specified
        if tmpdir is None:
            MAX_LEN = 512
            # 32 is whitespace
            dir_tensor = torch.full((MAX_LEN, ),
                                    32,
                                    dtype=torch.uint8,
                                    device='cuda')
            if rank == 0:
                mmcv.mkdir_or_exist('.dist_test')
                tmpdir = tempfile.mkdtemp(dir='.dist_test')
                tmpdir = torch.tensor(
                    bytearray(tmpdir.encode()),
                    dtype=torch.uint8,
                    device='cuda')
                dir_tensor[:len(tmpdir)] = tmpdir
            dist.broadcast(dir_tensor, 0)
            tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
        else:
            tmpdir = osp.join(tmpdir, '.dist_test')
            mmcv.mkdir_or_exist(tmpdir)
        # synchronizes all processes to make sure tmpdir exist
        dist.barrier()
        # dump the part result to the dir
        mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
        # synchronizes all processes for loading pickle file
        dist.barrier()
        # collect all parts
        if rank != 0:
            return None
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results

    def collect_results_gpu(result_part, size):  # noqa: F811
        """Collect results in gpu mode.

        It encodes results to gpu tensors and use gpu communication for results
        collection.

        Args:
            result_part (list): Results to be collected
            size (int): Result size.

        Returns:
            list: Ordered results.
        """
        rank, world_size = get_dist_info()
        # dump result part to tensor with pickle
        part_tensor = torch.tensor(
            bytearray(pickle.dumps(result_part)),
            dtype=torch.uint8,
            device='cuda')
        # gather all result part tensor shape
        shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
        shape_list = [shape_tensor.clone() for _ in range(world_size)]
        dist.all_gather(shape_list, shape_tensor)
        # padding result part tensor to max length
        shape_max = torch.tensor(shape_list).max()
        part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
        part_send[:shape_tensor[0]] = part_tensor
        part_recv_list = [
            part_tensor.new_zeros(shape_max) for _ in range(world_size)
        ]
        # gather all result part
        dist.all_gather(part_recv_list, part_send)

        if rank == 0:
            part_list = []
            for recv, shape in zip(part_recv_list, shape_list):
                part_list.append(
                    pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
            # sort the results
            ordered_results = []
            for res in zip(*part_list):
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            ordered_results = ordered_results[:size]
            return ordered_results
        return None
