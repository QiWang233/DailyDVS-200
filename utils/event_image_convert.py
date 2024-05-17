# Read .aedat4 file from DAVIS346 at the temporal window (t1-t2), generating frames.

import os
import numpy as np
import math
from dv import AedatFile
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import json
import tqdm as tq
from concurrent.futures import ThreadPoolExecutor
import random
from numpy.lib import recfunctions
import shutil

class EventImageConvert:

    def __init__(self, width=320, height=240, interval=0.25, class_num=200, output_path=None) :
        self.width = width
        self.height = height
        
        self.interval = interval
        self.output_path = output_path

        self.class_num = class_num

    def show_events(self,events):
        """
        plot events in three-dimensional space.

        Inputs:
        -------
            true events   - the true event signal.
            events   - events include true event signal and noise events.
            width    - the width of AER sensor.
            height   - the height of AER sensor.

        Outputs:
        ------
            figure     - a figure shows events in three-dimensional space.

        """
        ON_index = np.where(events['polarity'] == 1)
        OFF_index = np.where(events['polarity'] == 0)

        
        fig = plt.figure('{} * {}'.format(self.width, self.height))
        ax = fig.gca(projection='3d')

        ax.scatter(events['timestamp'][ON_index]-events['timestamp'][0], events['x'][ON_index], events['y'][ON_index], c='red', label='ON', s=3)  # events_ON[1][:]
        ax.scatter(events['timestamp'][OFF_index]-events['timestamp'][0], events['x'][OFF_index], events['y'][OFF_index], c='mediumblue', label='OFF', s=3)

        font1 = {'family': 'Times New Roman', 'size': 20}
        font1_x = {'family': 'Times New Roman', 'size': 19}
        font2 = {'size': 13}
        ax.set_xlabel('t(us)', font1_x)  # us 
        ax.set_ylabel('x', font1)
        ax.set_zlabel('y', font1)
        plt.show()

    def make_color_histo(self,events, img=None):
        """
        simple display function that shows negative events as blue dots and positive as red one
        on a white background
        args :
            - events structured numpy array: timestamp, x, y, polarity.
            - img (numpy array, height x width x 3) optional array to paint event on.
            - width int.
            - height int.
        return:
            - img numpy array, height x width x 3).
        """

        if img is None:
            img = 255 * np.ones((self.height, self.width, 3), dtype=np.uint8)
        else:
            # if an array was already allocated just paint it grey
            img[...] = 255
        if events.size:
            
            assert events['x'].max() < self.width, "out of bound events: x = {}, w = {}".format(events['x'].max(), self.width)
            assert events['y'].max() < self.height, "out of bound events: y = {}, h = {}".format(events['y'].max(), self.height)

            ON_index = np.where(events['polarity'] == 1)
            img[events['y'][ON_index], events['x'][ON_index], :] = [30, 30, 220] * events['polarity'][ON_index][:, None]  # red
            OFF_index = np.where(events['polarity'] == 0)
            img[events['y'][OFF_index], events['x'][OFF_index], :] = [200, 30, 30] * (events['polarity'][OFF_index] + 1)[:,None]  # blue
        return img

    def _get_frames_NUM(self, input_filename):
        """
            Get the frame count for each event data
        """
        
        with AedatFile(input_filename) as f:
            events = np.hstack([event for event in f['events'].numpy()])
            timestamps = [t[0] for t in events]
            return math.ceil( ( max(timestamps) - min(timestamps) ) * 1e-6 / self.interval )

    def _events_to_event_images(self,input_filename, output_file, relative_path, aps_frames_NUM, interval, label):
        """
        Mapping asynchronous events into event images
        args :
            - input_file:.aedat file, saving dvs events.
            - output_file: the output filename saving timestamps.
            - relative_path : relative path of png image with respect to the root path 
            - aps_frames_NUM: the number of the event data.
            - interval:time interval
            - label:the label of action
        return:
            - event_image
            - txt:saving the name of timestamp,frames_num,label
        """

        if os.path.exists(input_filename):
        
            with AedatFile(input_filename) as f:
                events = np.hstack([event for event in f['events'].numpy()])

                start_timestamp = events[0][0] 

                # saving event images.
                for i in range(int(aps_frames_NUM)):     
                   
                    start_index = np.searchsorted(events['timestamp'], int(start_timestamp)+i*interval*1e6) 
                    end_index = np.searchsorted(events['timestamp'], int(start_timestamp)+(i+1)*interval*1e6)

                    print("start_index=",start_index)
                    print("end_index=",end_index)

                    rec_events = events[start_index:end_index]
                    print(rec_events)

                    # print("rec_events:",type(rec_events))

                    event_image = self.make_color_histo(rec_events)
                    save_path = output_file +'/{:08d}.png'.format(i)

                    # print(save_path)
                    cv2.imwrite(save_path, event_image)

                    # print('The filename {}, the {} frame has been done!'.format(input_filename, i+1))
                    # output_filename.write(relative_path+ "_dvs" + " {}".format(aps_frames_NUM) + " {}".format(label) + '\n')  # save the timestamp,frames_NUM,label
            
    
