# ==============================================================================
# Build dataset and dataloader with augmentations for training and testing.
# More details please see: https://github.com/fangwei123456/spikingjelly.git
# Papers: GET: Group Event Transformer for Event-Based Vision, ICCV 2023.
# ==============================================================================

import os
import numpy as np
import random
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import pdb
import matplotlib.pyplot as plt
from data.mixup_custom import Mixup

from aermanager.preprocess import crop_events

from spikingjelly.datasets import play_frame
from spikingjelly.datasets import split_to_train_test_set
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS as jelly_CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture as jelly_DVS128Gesture
from spikingjelly.datasets.our_dvs import OurDVS as jelly_OurDVS
from spikingjelly.datasets.n_caltech101 import NCaltech101 as jelly_NCaltech101
import time

from event_based.event_token import E2IMG, save_tensor_as_image
try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_loader(config):
    config.defrost()
    print(config.DATA.DATA_PATH)
    if 'dvs128gesture' in config.DATA.DATA_PATH:
        os.makedirs(config.DATA.DATA_PATH, exist_ok=True)
        dataset_train = DVS128Gesture(root=config.DATA.DATA_PATH, train=True)
        dataset_val = DVS128Gesture(root=config.DATA.DATA_PATH, train=False)
        config.MODEL.NUM_CLASSES = 11
    elif 'cifar10dvs' in config.DATA.DATA_PATH:
        os.makedirs(config.DATA.DATA_PATH, exist_ok=True)
        dataset_trainval = split_to_train_test_set(0.9, jelly_CIFAR10DVS(root=config.DATA.DATA_PATH), 10)
        # pdb.set_trace()
        dataset_train = CIFAR10DVS(dataset_trainval, train=True)
        dataset_val = CIFAR10DVS(dataset_trainval, train=False)       
        config.MODEL.NUM_CLASSES = 10
    elif 'our_dvs' in config.DATA.DATA_PATH:
        os.makedirs(config.DATA.DATA_PATH, exist_ok=True)
        dataset_train = OurDVS(root=config.DATA.DATA_PATH, train=True)
        dataset_val = OurDVS(root=config.DATA.DATA_PATH, train=False)
        config.MODEL.NUM_CLASSES = 200
    elif 'NCaltech101' in config.DATA.DATA_PATH:
        os.makedirs(config.DATA.DATA_PATH, exist_ok=True)
        dataset_train = NCaltech101(root=config.DATA.DATA_PATH, train=True)
        dataset_val = NCaltech101(root=config.DATA.DATA_PATH, train=False)
        config.MODEL.NUM_CLASSES = 101
    config.freeze()

    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build datasets")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
   
    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )
        
    print(dataset_train.__getitem__(0))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True
    )
    # print(data_loader_train[0])
    # config.DATA.DATASET
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val,
        batch_size=2 * config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        collate_fn=custom_collate_fn,
        persistent_workers=True
    )

    # setup mixup / cutmix
    mixup_fn = None
    print(config.AUG.MIXUP,config.AUG.CUTMIX,config.AUG.CUTMIX_MINMAX)
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    print(f'a:{mixup_active}')
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


class THU_EACT_50_CHL:
    def __init__(self, datafile="../THU-EACT-50-CHL", eval=False, augmentation=False,
                 repr=['timeSurface'], time_num=9, ret_file_name=False, demo=False):
        list_file_name = join(datafile,"test.txt") if eval else join(datafile,"train.txt")
        if demo:
            list_file_name = join(datafile, "test-demo.txt") if eval else join(datafile, "train-demo.txt")

        self.files = []
        self.labels = []
        self.augmentation = augmentation
        self.datafile = datafile

        self.repr = repr
        self.time_num = time_num
        self.ret_file_name = ret_file_name

        list_file = open(list_file_name, "r")
        for line in list_file:
            file, label = line.split(" ")
            self.files.append(file)
            self.labels.append(int(label))
        list_file.close()

        self.classes = np.unique(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        f = self.files[idx]
        f = f.split('DVS-action-data-npy/')[-1]
        f = os.path.join(self.datafile, f)

        events = np.load(f).astype(np.float32)

        # normalize the timestamps
        _min = events[:,2].min()
        _max = events[:,2].max()
        events[:,2] = (events[:,2] - _min) / (_max - _min)

        if self.augmentation:
            events = random_shift_events(events)
            events = random_flip_events_along_x(events)

        reprs = []
        for repr_name in self.repr:
            repr_array = repr_map[repr_name](events[:, 2], events[:, 0].astype(np.int32), events[:, 1].astype(np.int32),
                                             events[:, 3],
                                             repr_size=(260, 346), time_num=self.time_num)

            # standardization
            # mu = np.mean(repr_array)
            # sigma = np.std(repr_array)
            # repr_array = (repr_array - mu) / sigma

            reprs.append(repr_array)

        reprs = np.array(reprs)
        if self.ret_file_name:
            # file_name = re.findall(r'A[\w-]+', f)[0]
            file_name = f.split('/')[-1].split('.')[0]
            return reprs, label, file_name
        else:
            return reprs, label


class CIFAR10DVS(Dataset):
    def __init__(self, jelly_dataset, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.jelly_CIFAR10DVS = jelly_dataset[0] if train else jelly_dataset[1]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (events, target) where target is index of the target class.
        """
        # pdb.set_trace()
        events, target = self.jelly_CIFAR10DVS.__getitem__(index)
        t = torch.tensor(events['t'].astype(np.float32)).unsqueeze(1)
        x = torch.tensor(events['x'].astype(np.float32)).unsqueeze(1)
        y = torch.tensor(events['y'].astype(np.float32)).unsqueeze(1)
        p = torch.tensor(events['p'].astype(np.float32)).unsqueeze(1)
        events = torch.cat([t, x, y, p], dim=1)
        print(events)
        H, W = (128, 128)
        if self.train:
            available_augmentations = \
            [flip_events, random_crop]
        
            weights = [30, 50, 20]
            choices = [0, 1, 2]
            k = random.choices(choices, weights)[0]
            selected_augmentations = random.sample(available_augmentations, k=k)
            for augmentation in selected_augmentations:
                events = augmentation(events)
            if k != 0:
                events[:, 1].clamp_(min=0, max=W - 1)
                events[:, 2].clamp_(min=0, max=H - 1)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return events, torch.tensor(target).long().squeeze(-1)

    def __len__(self):
        return self.jelly_CIFAR10DVS.__len__()

# (302, 245)
class NCaltech101(Dataset):
    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)
        self.train = train
        self.jelly_DVS128Gesture = jelly_NCaltech101(root, train)
        self.e2i = E2IMG([240, 180])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (events, target) where target is index of the target class.
        """
        t0 = time.time()
        H, W = (180, 240)
        events, target = self.jelly_NCaltech101.__getitem__(index)
        # print(events['t'])
        if len(events['t'])>self.max_point:
            bias = random.randint(0, len(events['t'])-self.max_point)
            # events[bias:bias+self.max_point]
            t = torch.tensor(events['t'][bias:bias+self.max_point]).unsqueeze(1)
            x = torch.tensor(events['x'][bias:bias+self.max_point]).unsqueeze(1)
            y = torch.tensor(events['y'][bias:bias+self.max_point]).unsqueeze(1)
            p = torch.tensor(events['p'][bias:bias+self.max_point]).unsqueeze(1)
        else:
            t = torch.tensor(events['t']).unsqueeze(1)
            x = torch.tensor(events['x']).unsqueeze(1)
            y = torch.tensor(events['y']).unsqueeze(1)
            p = torch.tensor(events['p']).unsqueeze(1)

        events = torch.cat([t, x, y, p], dim=1) / 1.0

        if self.train:
            available_augmentations = [dummy, random_crop2]
           
            weights = [25, 25, 50]
            choices = [0, 1, 2]
            k = random.choices(choices, weights)[0]
            # k = 2
            selected_augmentations = [available_augmentations[k]]
            if k == 2 and random.random() > 0.5:
                selected_augmentations.append(flip_events)
            for augmentation in selected_augmentations:
                if augmentation == random_crop2:
                    events = augmentation(events, input_size=(H, H))
                else:
                    events = augmentation(events)
            if k != 0:
                events[:, 1].clamp_(min=0, max=W - 1)
                events[:, 2].clamp_(min=0, max=H - 1)

        sorted_indices = torch.argsort(events[:, 0])
        events = events[sorted_indices]

        if self.target_transform is not None:
            target = self.target_transform(target)

        # print(events.shape,torch.tensor(target).long().squeeze(-1))
        return events, torch.tensor(target).long().squeeze(-1)

    def __len__(self):
        return self.jelly_NCaltech101.__len__()

class DVS128Gesture(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.jelly_DVS128Gesture = jelly_DVS128Gesture(root, train)
        self.e2i = E2IMG([128, 128])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (events, target) where target is index of the target class.
        """

        events, target = self.jelly_DVS128Gesture.__getitem__(index)
        t = torch.tensor(events['t']).unsqueeze(1)
        x = torch.tensor(events['x']).unsqueeze(1)
        y = torch.tensor(events['y']).unsqueeze(1)
        p = torch.tensor(events['p']).unsqueeze(1)
        events = torch.cat([t, x, y, p], dim=1) / 1.0
        H, W = (128, 128)
        if self.train:
            available_augmentations = \
            [dummy, flip_events, random_crop]
            weights = [25, 25, 50]
            choices = [0, 1, 2]
            k = random.choices(choices, weights)[0]
            selected_augmentations = [available_augmentations[k]]
            if k == 2 and random.random() > 0.5:
                selected_augmentations.append(flip_events)
            for augmentation in selected_augmentations:
                events = augmentation(events)
            if k != 0:
                events[:, 1].clamp_(min=0, max=W - 1)
                events[:, 2].clamp_(min=0, max=H - 1)
            if flip_events in selected_augmentations:
                target_mapping = {1: 3, 3: 1, 5: 6,
                                  6: 5, 4: 7, 7: 4}
                if target in target_mapping:
                    target = target_mapping[target]

        sorted_indices = torch.argsort(events[:, 0])
        events = events[sorted_indices]

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # print(events.shape,torch.tensor(target).long().squeeze(-1))
        return events, torch.tensor(target).long().squeeze(-1)

    def __len__(self):
        return self.jelly_DVS128Gesture.__len__()

class OurDVS(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        # print(self.root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.jelly_OurDVS = jelly_OurDVS(root, train)
        self.e2i = E2IMG([320, 240])
        self.max_point = 500000
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (events, target) where target is index of the target class.
        """
        t0 = time.time()
        H, W = (240, 320)
        events, target = self.jelly_OurDVS.__getitem__(index)
        # print(events['t'])
        if len(events['t'])>self.max_point:
            bias = random.randint(0, len(events['t'])-self.max_point)
            # events[bias:bias+self.max_point]
            t = torch.tensor(events['t'][bias:bias+self.max_point]).unsqueeze(1)
            x = torch.tensor(events['x'][bias:bias+self.max_point]).unsqueeze(1)
            y = torch.tensor(events['y'][bias:bias+self.max_point]).unsqueeze(1)
            p = torch.tensor(events['p'][bias:bias+self.max_point]).unsqueeze(1)
        else:
            t = torch.tensor(events['t']).unsqueeze(1)
            x = torch.tensor(events['x']).unsqueeze(1)
            y = torch.tensor(events['y']).unsqueeze(1)
            p = torch.tensor(events['p']).unsqueeze(1)

        events = torch.cat([t, x, y, p], dim=1) / 1.0

        if self.train:
            available_augmentations = [dummy, flip_events, random_crop2]
           
            weights = [25, 25, 50]
            choices = [0, 1, 2]
            k = random.choices(choices, weights)[0]
            # k = 2
            selected_augmentations = [available_augmentations[k]]
            if k == 2 and random.random() > 0.5:
                selected_augmentations.append(flip_events)
            for augmentation in selected_augmentations:
                if augmentation == flip_events:
                    events = augmentation(events, W=W)
                elif augmentation == random_crop2:
                    events = augmentation(events, input_size=(H, H))
                else:
                    events = augmentation(events)
            if k != 0:
                events[:, 1].clamp_(min=0, max=W - 1)
                events[:, 2].clamp_(min=0, max=H - 1)

        sorted_indices = torch.argsort(events[:, 0])
        events = events[sorted_indices]

        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, torch.tensor(target).long().squeeze(-1)

    def __len__(self):
        return self.jelly_OurDVS.__len__()
    

def events_to_frames(events, width, height, duration, frame_interval):
    
    num_frames = int(duration / frame_interval)

    frames_z = np.zeros((num_frames, height, width), dtype=np.float32)
    frames_f = np.zeros((num_frames, height, width), dtype=np.float32)

    for event in events:
        t, x, y, p = event.numpy()
        frame_index = int(t / frame_interval)
        if(frame_index==num_frames):
            break
        if int(p)==0:
            frames_f[frame_index, int(y), int(x)] = 1
        if int(p)>0:
            frames_z[frame_index, int(y), int(x)] = 1
    return frames_z ,frames_f

def visualize_eventframe(frames,dir,ps='zheng'):

    
    for i, frame in enumerate(frames):
        # print(frame)
        # print(sum(frame))
        
        plt.imshow(frame, cmap='bwr', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title(f"Event Frame{i}")
        file_path = os.path.join(dir,f'figure_{ps}_{i}.png')
        print(file_path)
        plt.savefig(file_path)
        plt.clf()

import concurrent.futures
def color_mapping(p):
    if int(p) == 1:
        return 'b'
    else:
        return 'r'

def plot_and_save_polarity_colored_3d_point_cloud(events, 
                                                  resolution, 
                                                  output_file='./visualization/output_polarity_colored_3d_point_cloud1.png'):
    print('start')
    x = events[:,1]
    y = events[:,2]
    t = events[:,0]
    polarity = events[:,3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = []


    with concurrent.futures.ThreadPoolExecutor(6) as executor:
        colors = list(executor.map(color_mapping, polarity))

    ax.scatter(t, x, y, s=1, c=colors)

    ax.set_xlabel('Timestamp')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('X')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    ax.set_zlim(resolution[1], 0)
    ax.set_ylim(-2*resolution[0], 2*resolution[0])

    ax.set_xlim(min(t), max(t))

    plt.savefig(output_file)
    plt.close()
    print('done draw')
#################################################################################
    
def crop_events():
    left, right = 35,35
    top, bottom = 4, 4

    # events_cropped, new_shape = crop_events(
    #     events,
    #     input_shape=shape,
    #     crop_size=((left, right), (top, bottom)))

    # new_shape

def dummy(events):
    """
    Do nothing.
    """    
    return events


def flip_events(events, W=127):
    """
    Flip events horizontally.
    """

    # print(events[0:10, 1])
    events[:, 1] = W - events[:, 1]
    # print(events[0:10, 1])
    return events

def random_crop2(events, spatial_crop_range=[0.7, 1.0], time_crop_range=[0.6, 1.0], input_size=(128, 128)):
    """
    Randomly crop events in space and time.
    """
    min_x, max_x = 0, input_size[1]
    min_y, max_y = 0, input_size[0]
    min_t, max_t = int(events[0, 0]), int(events[-1, 0])

    if random.random() >= 0.5:
        # Spatial cropping
        scale = torch.rand(2) * (spatial_crop_range[1] - spatial_crop_range[0]) + spatial_crop_range[0]
        crop_size_x = int(scale[0] * (max_x - min_x))
        crop_size_y = int(scale[1] * (max_y - min_y))
        start_x = int(torch.randint(0, max_x - crop_size_x, (1,)))
        start_y = int(torch.randint(0, max_y - crop_size_y, (1,)))
        mask_x = torch.logical_and(events[:, 1] >= start_x, events[:, 1] <= start_x + crop_size_x)
        mask_y = torch.logical_and(events[:, 2] >= start_y, events[:, 2] <= start_y + crop_size_y)
        crop_mask = torch.logical_and(mask_x, mask_y)
        cropped_events = events[crop_mask]

        # Adaptive shift based on crop size
        x_shift = torch.randint(-start_x, input_size[1] - start_x - crop_size_x + 1, size=(1,))
        y_shift = torch.randint(-start_y, input_size[0] - start_y - crop_size_y + 1, size=(1,))
        cropped_events[:, 1] += x_shift
        cropped_events[:, 2] += y_shift
    else:
        # Time cropping
        time_crop_range[1] = (max_t - min_t - 1) / (max_t - min_t)
        scale = torch.rand(1) * (time_crop_range[1] - time_crop_range[0]) + time_crop_range[0]
        crop_size_t = int(scale * (max_t - min_t))
        start_t = int(torch.randint(min_t, max_t - crop_size_t, (1,)))
        crop_mask = torch.logical_and(events[:, 0] >= start_t, events[:, 0] <= start_t + crop_size_t)
        cropped_events = events[crop_mask]

    return cropped_events


def random_crop(events, spatial_crop_range=[0.7, 126 / 127], time_crop_range=[0.6, 1.0]):
    """
    Randomly crop events in space and time.
    """
    min_x, max_x = 0, 127
    min_y, max_y = 0, 127
    min_t, max_t = int(events[0, 0]),  int(events[-1, 0])
    # print(max_t, min_t)
    if random.random() >= 0.5:
        # Spatial cropping
        scale = torch.rand(2) * (spatial_crop_range[1] - spatial_crop_range[0]) + spatial_crop_range[0]
        crop_size_x = int(scale[0] * (max_x - min_x))
        crop_size_y = int(scale[1] * (max_y - min_y))
        start_x = int(torch.randint(0, max_x - crop_size_x, (1,)))
        start_y = int(torch.randint(0, max_y - crop_size_y, (1,)))
        mask_x = torch.logical_and(events[:, 1] >= start_x, events[:, 1] <= start_x + crop_size_x)
        mask_y = torch.logical_and(events[:, 2] >= start_y, events[:, 2] <= start_y + crop_size_y)
        crop_mask = torch.logical_and(mask_x, mask_y)
        cropped_events = events[crop_mask]

        # Adaptive shift based on crop size
        x_shift = torch.randint(-start_x, 127 - start_x - crop_size_x + 1, size=(1,))
        y_shift = torch.randint(-start_y, 127 - start_y - crop_size_y + 1, size=(1,))
        cropped_events[:, 1] += x_shift
        cropped_events[:, 2] += y_shift    

    else:
        # Time cropping
        time_crop_range[1] = (max_t - min_t - 1) / (max_t - min_t)
        scale = torch.rand(1) * (time_crop_range[1] - time_crop_range[0]) + time_crop_range[0]
        crop_size_t = int(scale * (max_t - min_t))
        start_t = int(torch.randint(min_t, max_t - crop_size_t, (1,)))
        crop_mask = torch.logical_and(events[:, 0] >= start_t, events[:, 0] <= start_t + crop_size_t)
        cropped_events = events[crop_mask]
    return cropped_events


def xy_shift_events(events, max_shift=20):
    """
    Randomly shift events in x and y directions. (not used)
    """
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2,))
    events[:, 1] += x_shift
    events[:, 2] += y_shift
    return events


def time_shift_events(events, max_shift=500):
    """
    Randomly shift events in time. (not used)
    """      
    time_shift = np.random.uniform(low=-max_shift, high=max_shift, size=len(events))
    events[:, 0] += time_shift
    sorted_indices = torch.argsort(events[:, 0])
    events = events[sorted_indices]
    return events


def polarity_inversion(events):
    """
    Randomly invert polarities. (not used)
    """
    events[:, 3] = 1 - events[:, 3]
    return events


def random_time_sampling(events, min_ratio=0.6, max_ratio=1.5):
    """
    Randomly sample events in time. (not used)
    """
    ratio = np.random.uniform(min_ratio, max_ratio)
    events = torch.nn.functional.interpolate(events.T.unsqueeze(0), scale_factor=ratio, mode='nearest')[0].T
    return events


def custom_collate_fn(batch):
    events_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    labels = torch.stack(labels_list, 0)
    return events_list, labels

