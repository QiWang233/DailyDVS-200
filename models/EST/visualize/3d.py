import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dv import AedatFile

def plot_and_save_polarity_colored_3d_point_cloud(events, resolution, output_file):
    x = events['x']
    y = events['y']
    t = events['timestamp']
    polarity = events['polarity']


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = []

    for p in polarity:
        if p==True:
            colors.append('b')
        else:
            colors.append('r')

    ax.scatter(t, x, y, s=1, c=colors) 

    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')

    ax.set_zlim(resolution[1],0)
    ax.set_ylim(0, resolution[0])
    ax.set_xlim(min(t), max(t))

    plt.savefig(output_file)
    plt.close()

def read_aedat_file(file_path):
    with AedatFile(file_path) as f:
        events = {'x': [], 'y': [], 'timestamp': [], 'polarity': []}

        for event in f['events']:
            events['x'].append(event.x)
            events['y'].append(event.y)
            events['timestamp'].append(event.timestamp)
            events['polarity'].append(event.polarity)

    return events



if __name__ == '__main__':
    aedat_file_path = 'dataset/C0P2M0S1_20231107_15_25_24.aedat4'

    events_data = read_aedat_file(aedat_file_path)

    
