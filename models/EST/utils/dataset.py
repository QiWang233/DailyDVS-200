import numpy as np
from os import listdir
from os.path import join
import time
import random

def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))
    events[:,0] += x_shift
    events[:,1] += y_shift

    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]

    return events

def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:,0] = W - 1 - events[:,0]
    return events

# (302, 245)
class NCaltech101:
    def __init__(self, root, augmentation=False):
        self.classes = listdir(root)

        # print(self.classes)
        self.files = []
        self.labels = []

        self.augmentation = augmentation

        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)
        # print(self.files)
        # print(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """

        t0 = time.time()
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)
        # print(events)
        # exit()
        '''
        [[ 1.20000e+02  2.00000e+00  5.00000e-06  1.00000e+00]
        [ 6.50000e+01  1.22000e+02  7.90000e-05 -1.00000e+00]
        '''
        if self.augmentation:
            events = random_shift_events(events)
            events = random_flip_events_along_x(events)
        

        # print(time.time()-t0, f, len(events))
        return events, label


class Ours:
    def __init__(self, root, augmentation=False):
        # self.classes = listdir(root)
        self.classes = sorted(listdir(root))
        # print(self.classes)
        self.max_point = 500000
        self.files = []
        self.labels = []

        self.augmentation = augmentation

        for i, c in enumerate(self.classes):
            # print(f"i:{i} c:{c}")
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            # new_files = [join(root, c, f) for f in listdir(join(root, c)) if 'M1' not in f]
            # print(new_files)
            # action = new_files[i].split("/")[-2]
            # print(f"c:{c.split('_')[-1]}")
            self.files += new_files
            # self.labels += [i] * len(new_files)
            # print(f"i:{i} len:{len(new_files)}")

            if c.split('_')[-1] != "download":
                self.labels += [int(c.split("_")[-1]) - 1] * len(new_files)

        # print(f"size:{len(self.labels)}")
        # for i,f in enumerate(self.files):
        #     print(f"f:{f} lable:{self.labels[i]}")
        # print(self.files)
        # print(self.labels)
        # print(self.labels)
    def read_npz_for_rpg(self, file_path):
        t0 = time.time()
        data = np.load(file_path)
       
        # if len(data['t'])>self.max_point:
        #     start = random.randint(0,  len(data['t'])-self.max_point)
        #     end = start+self.max_point-1
        # else:
        #     start = 0
        #     end = len(data['t'])-1

        t1 = time.time()
        # print(data['t'][0])
        # a = [int(t) - int(data['t'][0]) for t in data['t']]
        a = data['t'] - data['t'][0]
        # print(a)

        events = np.column_stack((data['x'],data['y'], a, data['p']))
        # events = np.vstack((data['x'], data['y'], data['t'], data['p'])).T
        # events = np.c_[data['x'], data['y'], a, data['p']]
        # exit()

        # print(t1-t0, time.time()-t1, file_path, len(events))
        
        return events.astype(np.float32)
        
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
        # print(label, f)
        
        if f.split('.')[1]=='npz':
            events = self.read_npz_for_rpg(f)
        elif f.split('.')[1]=='npy':
            t0 = time.time()
            events = np.load(f).astype(np.float32)
            t1 = time.time()
            
        # events = events[events[:,3]!=0.]
        # print(events)
        # exit()
        #  normalize the timestamps
        _min = events[:,2].min()
        _max = events[:,2].max()
        events[:,2] = (events[:,2] - _min) / (_max - _min)

        if self.augmentation:
            cut_flag = random.randint(1,10)
            if cut_flag >= 4 and len(events)>self.max_point:
                bias = random.randint(0, (len(events)-self.max_point))
                events = events[bias:bias+self.max_point]
            events = random_shift_events(events, resolution=(240, 320))
            events = random_flip_events_along_x(events, resolution=(240, 320))
        # print(f'label:{label}')
        # print(t1-t0, time.time()-t1, len(events))
        return events, label