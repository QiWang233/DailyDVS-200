import numpy as np
import os
import re
from os import listdir
from os.path import join
from scipy import io
import pandas as pd
# from torch.utils.data import DataLoader, Dataset
from process import *

repr_map = {'eventFrame':get_eventFrame,
            'eventAccuFrame':get_eventAccuFrame,
            'timeSurface':get_timeSurface,
            'eventCount':get_eventCount}

# left or right move all event locations randomly
def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))
    events[:,0] += x_shift
    events[:,1] += y_shift

    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]

    return events

# flip half of the event images along the x dimension
def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:,0] = W - 1 - events[:,0]
    return events



class DHP19:
    def __init__(self, datafile="../DHP19", eval=False, augmentation=False, camera_id=3,
                 repr=['timeSurface'], time_num=9):
        list_file_name = join(datafile,"test.txt") if eval else join(datafile,"train.txt")

        self.files = []
        self.labels = []
        self.augmentation = augmentation
        self.camera_id = camera_id

        self.repr = repr
        self.time_num = time_num

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
        whole_events = io.loadmat(f)['events'].astype(np.float32)

        # Important for DHP19
        # choose the camera_id for training and testing
        events = whole_events[whole_events[:, -1] == self.camera_id][:,:-1]

        # normalize the timestamps
        _min = events[:,2].min()
        _max = events[:,2].max()
        events[:,2] = (events[:,2] - _min) / (_max - _min)

        # change the original (x.y) ([1,346],[1,260]) to ([0,345],[0,259])
        events[:, 0] = events[:, 0] - 1
        events[:, 1] = events[:, 1] - 1

        if self.augmentation:
            events = random_shift_events(events)
            events = random_flip_events_along_x(events)

        # return events, label

        reprs = []
        for repr_name in self.repr:
            repr_array = repr_map[repr_name](events[:, 2], events[:, 0].astype(np.int32), events[:, 1].astype(np.int32), events[:, 3],
                            repr_size=(260, 346), time_num=self.time_num)

            # standardization
            # mu = np.mean(repr_array)
            # sigma = np.std(repr_array)
            # repr_array = (repr_array - mu) / sigma

            reprs.append(repr_array)

        reprs = np.array(reprs)
        return reprs, label


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


class THU_EACT_50:
    def __init__(self, datafile="../THU_EACT_50", mode="front", eval=False, augmentation=False, max_points=1000000,
                 repr=['timeSurface'], time_num=9):
        list_file_name = None
        if mode == "front": # front views (C1-C2)
            list_file_name = join(datafile,"test.txt") if eval else join(datafile,"train.txt")
        elif mode.startswith("view_"): # just a single view
            list_file_name = join(datafile, "test_" + mode + ".txt") if eval else join(datafile, "train_" + mode + ".txt")

        self.files = []
        self.labels = []
        self.augmentation = augmentation
        self.max_points = max_points
        self.datafile = datafile

        self.repr = repr
        self.time_num = time_num

        list_file = open(list_file_name, "r")
        for line in list_file:
            file, label = line.split(",")
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
        f = os.path.join(self.datafile, self.files[idx])


        # read the raw csv data and calculate the representations
        pd_reader = pd.read_csv(f, header=None).values
        events = np.vstack((pd_reader[:, 1], pd_reader[:, 0], pd_reader[:, 4], pd_reader[:, 3])).T.astype(np.float32)
        events = events[events[:,3]!=0.] # delete all the points that have the polarity of 0

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
                                             events[:, 3], repr_size=(800, 1280), time_num=self.time_num)
            # standardization
            # mu = np.mean(repr_array)
            # sigma = np.std(repr_array)
            # repr_array = (repr_array - mu) / sigma

            reprs.append(repr_array)
        reprs = np.array(reprs)
        return reprs, label


if __name__ == '__main__':
    # for THU-EACT-50
    data_directory = "H:/Event_camera_action/THU-EACT-50"
    repr = ['timeSurface']
    dataset = THU_EACT_50(datafile=data_directory, mode="front", eval=True, augmentation=False, repr=repr)

    # for THU-EACT-50-CHL
    # data_directory = "H:/Event_camera_action/THU-EACT-50-CHL"
    # repr = ['timeSurface']
    # dataset = THU_EACT_50_CHL(datafile=data_directory, eval=True, augmentation=False, repr=repr)

    index_to_test = 0  # index of the sample you want to test
    single_sample_reprs, single_sample_label = dataset.__getitem__(index_to_test)

    # Output the results
    print("Representation Shape:", single_sample_reprs.shape)
    print("Label:", single_sample_label)