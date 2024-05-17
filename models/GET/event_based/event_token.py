# ==============================================================================
# Group Token Embedding.
# Copyright (c) 2023 The Group Event Transformer Authors.
# Licensed under The MIT License.
# Written by Yansong Peng.
# ==============================================================================

import torch
import torch.nn as nn
from PIL import Image


class E2SRC_Module(nn.Module):

    @torch.no_grad()
    def __init__(self, shape, group_num, patch_size):
        super().__init__()
        self.H, self.W = int(shape[1]) - 1, int(shape[0]) - 1
        self.time_div = group_num // 2
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

    def forward(self, x):  # Input: x → [N, 4] tensor, each row is (t, x, y, p).
        """
        Given a set of events, return event tokens.
        """
        # print(self.patch_size)
        # print('******* input token start********')
        # [t w h p]
        x = x[x != torch.inf].reshape(-1, 4)  # remove padding
        # print('************')
        # print(x)
        # print('***********')
        PH, PW = int((self.H + 1) / self.patch_size[0]), int((self.W + 1) / self.patch_size[1])
        # print(self.H, self.W)
        # print(PH, PW)

        Token_num, Patch_size, b = int(PH * PW), int(self.patch_size[0] * self.patch_size[1]), 1e-4
        # self.time_div = 6  [6,2,2,4*4, H/4 * W/4]
        y = torch.zeros([self.time_div, 2, 2, Patch_size, Token_num], dtype=x[0].dtype, device=x[0].device)
        if len(x):
            w = x[:, 3] != 2
            # (t - t0)/（t_end - t_0） relative_time
            wt = torch.div(x[:, 0] - x[0, 0], x[-1, 0] - x[0, 0] + 1e-4)
            # (0, W) -> (0, PW - 1)
            Position = torch.div(x[:, 1], (self.W / PW + b), rounding_mode='floor') + \
                       torch.div(x[:, 2], (self.H / PH + b), rounding_mode='floor') * PW
            Token = torch.floor(x[:, 1] % (self.W / PW + 1e-4)) + \
                    torch.floor(x[:, 2] % (self.H / PH + b)) * int((self.W + 1) / PW)
            # print(Token, Token.size)
            t_double = x[:, 0].double()
            DTime = torch.floor(self.time_div * torch.div(t_double - t_double[0], t_double[-1] - t_double[0] + 1))

            # Mapping from 4-D to 1-D.
            bins = torch.as_tensor((self.time_div, 2, Patch_size, Token_num)).to(x.device)
            x_nd = torch.cat([DTime.unsqueeze(1), x[:, 3].unsqueeze(1), Token.unsqueeze(1), Position.unsqueeze(1)], dim=1).permute(1, 0).int()
            x_1d, index = index_mapping(x_nd, bins)

            # Get 1-D histogram which encodes the event tokens.
            # print('**************')
            # print(f"bins:{bins}")
            # print(f"index:{index}")
            # print(f"x_1d:{x_1d}, {len(x_1d)}")
            # print('**************')
            y[:, :, 0, :, :], y[:, :, 1, :, :] = get_repr(x_1d, index, bins=bins, weights=[w, wt])
        # print('******* input token end ********')
        return y.reshape(1, -1, PH, PW)  # Output: y → [1, group_num * '2' * (patch_size ** 2), H // patch_size, W // patch_size] tensor.
        
    # def forward(self, x):  
    #     """
    #     Given a set of events, return event tokens.
    #     """
    #     x = x[x != torch.inf].reshape(-1, 4)  # remove padding
        
    #     # Normalize coordinates to the range [0, 1]
    #     normalized_x = x[:, 1] / self.W
    #     normalized_y = x[:, 2] / self.H
        
    #     # Calculate relative time
    #     relative_time = (x[:, 0] - x[0, 0]) / (x[-1, 0] - x[0, 0] + 1e-4)

    #     # Map coordinates and time to tokens
    #     token_x = (normalized_x * self.patch_size[1]).floor().long()
    #     token_y = (normalized_y * self.patch_size[0]).floor().long()
    #     token_time = (relative_time * self.time_div).floor().long()

    #     # Calculate position in the 1-D representation
    #     Position = token_y * (self.patch_size[1] // self.time_div) + token_x

    #     # Create bins tensor
    #     bins = torch.as_tensor((self.time_div, 2, self.patch_size[0], self.patch_size[1])).to(x.device)

    #     # Concatenate features for mapping
    #     x_nd = torch.cat([token_time.unsqueeze(1), x[:, 3].unsqueeze(1), Position.unsqueeze(1)], dim=1).permute(1, 0).int()
    #     x_1d, index = index_mapping(x_nd, bins)

    #     # Get 1-D histogram which encodes the event tokens
    #     y = torch.zeros([self.time_div, 2, self.patch_size[0], self.patch_size[1]], dtype=x[0].dtype, device=x[0].device)
    #     w = x[:, 3] != 2
    #     wt = torch.div(x[:, 0] - x[0, 0], x[-1, 0] - x[0, 0] + 1e-4)
    #     y[:, 0, :, :], y[:, 1, :, :] = get_repr(x_1d, index, bins=bins, weights=[w, wt])

    #     return y.unsqueeze(0)  # Output: [1, time_div, 2, patch_size[0], patch_size[1]] tensor.



class E2SRC(nn.Module):

    def __init__(self, shape, batch_size=1, group_num=12, patch_size=4):
        super().__init__()
        # print(f"*************{shape}********************")
        self.module_list = nn.ModuleList([E2SRC_Module(shape, group_num, patch_size)] * batch_size)
 
    def forward(self, x):
        """
        Parallelly convert events into event tokens efficiently.
        """

        x_padded = torch.nn.utils.rnn.pad_sequence(x, padding_value=torch.inf).transpose(0, 1)
        y = torch.nn.parallel.parallel_apply(self.module_list[:len(x_padded)], x_padded)
        y = torch.cat(y, dim=0)

        return y
    

def index_mapping(sample, bins=None):
    """
    Multi-index mapping method from N-D to 1-D.
    """
    device = sample.device
    bins = torch.as_tensor(bins).to(device)
    y = torch.max(sample, torch.zeros([], device=device, dtype=torch.int32))
    y = torch.min(y, bins.reshape(-1, 1))
    index = torch.ones_like(bins)
    index[1:] = torch.cumprod(torch.flip(bins[1:], [0]), -1).int()
    index = torch.flip(index, [0])
    l = torch.sum((index.reshape(-1, 1)) * y, 0)
    return l, index


def get_repr(l, index, bins=None, weights=None):
    """
    Function to return histograms.
    """
    
    hist = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[0])
    hist = hist.reshape(tuple(bins))
    if len(weights) > 1:
        hist2 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[1])
        hist2 = hist2.reshape(tuple(bins))
    else:
        return hist
    if len(weights) > 2:
        hist3 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[2])
        hist3 = hist3.reshape(tuple(bins))
    else:
        return hist, hist2
    if len(weights) > 3:
        hist4 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[3])
        hist4 = hist4.reshape(tuple(bins))
    else:
        return hist, hist2, hist3
    return hist, hist2, hist3, hist4


class E2IMG(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        y = torch.stack([self.Ree(x[i]) for i in range(len(x))], dim=0)
        return y

    def Ree(self, x):
        """
        Convert events into images.
        """
        sz = self.args[0]
        y = 255 * torch.ones([3, int(sz[1]), int(sz[0])], dtype=x.dtype, device=x.device)
        if len(x):
            y[0, torch.floor(x[:, 2]).long(), torch.floor(x[:, 1]).long()] = 255 - 255 * (x[:, 3] == 1).to(dtype=y.dtype)
            y[1, torch.floor(x[:, 2]).long(), torch.floor(x[:, 1]).long()] = 255 - 255 * (x[:, 3] == 0).to(dtype=y.dtype)
            y[2] = y[0] + y[1]
        return y.permute(1, 2, 0)
    

def save_tensor_as_image(tensor, file_path):
    # Convert the tensor to a PIL image
    tensor = tensor.squeeze().detach().cpu()
    tensor = torch.clamp(tensor, 0, 1)
    tensor = (255 * tensor).to(torch.uint8)
    pil_image = Image.fromarray(tensor.numpy())
    pil_image.save(file_path)
