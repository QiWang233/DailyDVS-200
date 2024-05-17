import numpy as np
from scipy import io
import numba
import time
# from matplotlib import pyplot as plt

@numba.jit(nopython=True) # for accelerating
def get_eventFrame(ts, x, y, p, repr_size=(260,346), time_num=1):
    """
    get the event frame with multi time window of the events
    :param ts:
    :param x:
    :param y:
    :param p:
    :param repr_size:
    :param time_num: split how many windows in the temporal dimension
    :return: numpy with the shape (time_num,repr_size[0],repr_size[1])
    """

    img = np.zeros(shape=(time_num,repr_size[0],repr_size[1]), dtype=np.float32)

    # process each temporal window
    batch_bum = int(ts.size / time_num)
    for time_idx in range(time_num):
        # extract the corresponding info
        ts_part = ts[time_idx * batch_bum: (time_idx + 1) * batch_bum]
        p_part = p[time_idx * batch_bum: (time_idx + 1) * batch_bum]
        x_part = x[time_idx * batch_bum: (time_idx + 1) * batch_bum]
        y_part = y[time_idx * batch_bum: (time_idx + 1) * batch_bum]

        # change polarity to (0,1) if it is (1,-1)
        p_part = ((p_part + 1) / 2).astype(np.int32)

        for i in range(len(ts_part)):
            img[time_idx, y_part[i], x_part[i]] = (2.0 * p_part[i] - 1)

        # draw image
        # fig = plt.figure()
        # fig.suptitle('Event Frame')
        # plt.imshow(img[time_idx], cmap='gray')
        # plt.xlabel("x [pixels]")
        # plt.ylabel("y [pixels]")
        # plt.colorbar()
        # # plt.savefig('event_frame.jpg')
        # plt.show()
    return img

@numba.jit(nopython=True) # for accelerating
def get_eventCount(ts, x, y, p, repr_size=(260, 346), time_num=1):
    """
    get the event frame with multi time window of the events
    :param ts:
    :param x:
    :param y:
    :param p:
    :param repr_size:
    :param time_num: split how many windows in the temporal dimension
    :return: numpy with the shape (time_num, repr_size[0], repr_size[1])
    """

    img = np.zeros(shape=(time_num, repr_size[0], repr_size[1]), dtype=np.float32)

    # process each temporal window
    batch_bum = int(ts.size / time_num)
    for time_idx in range(time_num):
        # extract the corresponding info
        ts_part = ts[time_idx * batch_bum: (time_idx + 1) * batch_bum]
        p_part = p[time_idx * batch_bum: (time_idx + 1) * batch_bum]
        x_part = x[time_idx * batch_bum: (time_idx + 1) * batch_bum]
        y_part = y[time_idx * batch_bum: (time_idx + 1) * batch_bum]

        # change polarity to (0,1) if it is (1,-1)
        p_part = ((p_part + 1) / 2).astype(np.int32)

        # count number of points on each pixel
        pixel_counts = np.zeros(shape=(repr_size[0], repr_size[1]), dtype=np.int32)
        for i in range(len(ts_part)):
            pixel_counts[y_part[i], x_part[i]] += 1

        img[time_idx, :, :] = pixel_counts

    # normalize to (-1,1)
    img = 2 * ((img - img.min()) / (img.max() - img.min())) - 1
    return img


@numba.jit(nopython=True) # for accelerating
def get_eventAccuFrame(ts, x, y, p, repr_size=(260,346), time_num=1):
    """
    get the event accumulate frame with multi time window of the events
    :param ts:
    :param x:
    :param y:
    :param p:
    :param repr_size:
    :param time_num: split how many windows in the temporal dimension
    :return: numpy with the shape (time_num,repr_size[0],repr_size[1])
    """

    img = np.zeros(shape=(time_num,repr_size[0],repr_size[1]), dtype=np.float32)

    # process each temporal window
    batch_bum = int(ts.size / time_num)
    for time_idx in range(time_num):
        # extract the corresponding info
        ts_part = ts[time_idx * batch_bum: (time_idx + 1) * batch_bum]
        p_part = p[time_idx * batch_bum: (time_idx + 1) * batch_bum]
        x_part = x[time_idx * batch_bum: (time_idx + 1) * batch_bum]
        y_part = y[time_idx * batch_bum: (time_idx + 1) * batch_bum]

        # change polarity to (0,1) if it is (1,-1)
        p_part = ((p_part + 1) / 2).astype(np.int32)

        for i in range(len(ts_part)):
            img[time_idx, y_part[i], x_part[i]] += (2.0 * p_part[i] - 1)

        # draw image
        # fig = plt.figure()
        # fig.suptitle('Event Frame')
        # plt.imshow(img[time_idx], cmap='gray')
        # plt.xlabel("x [pixels]")
        # plt.ylabel("y [pixels]")
        # plt.colorbar()
        # # plt.savefig('event_frame.jpg')
        # plt.show()
    # normalize to (-1,1)
    # img = 2 * ((img - img.min()) / (img.max() - img.min())) - 1
    return img


@numba.jit(nopython=True) # for accelerating
def get_timeSurface(ts, x, y, p, repr_size=(260,346), time_num=1):
    """
    get the time surface with multi time window of the events
    :param ts:
    :param x:
    :param y:
    :param p:
    :param repr_size:
    :param time_num: split how many windows in the temporal dimension
    :return: numpy with the shape (time_num,repr_size[0],repr_size[1])
    """

    # parameters for Time Surface
    tau = 50e-3  # 50ms

    # sae = np.zeros(repr_size, np.float32)
    sae = np.zeros((time_num,repr_size[0],repr_size[1]), np.float32)

    # process each temporal window
    batch_bum = int(ts.size / time_num)
    for time_idx in range(time_num):
        # extract the corresponding info
        ts_part = ts[time_idx * batch_bum: (time_idx + 1) * batch_bum]
        p_part = p[time_idx * batch_bum: (time_idx + 1) * batch_bum]
        x_part = x[time_idx * batch_bum: (time_idx + 1) * batch_bum]
        y_part = y[time_idx * batch_bum: (time_idx + 1) * batch_bum]

        # calculate timesurface using expotential decay
        t_ref = ts_part[-1]  # 'current' time
        for i in range(len(ts_part)):
            if (p_part[i] > 0):
                sae[time_idx, y_part[i], x_part[i]] = np.exp(-(t_ref - ts_part[i]) / tau)
            else:
                sae[time_idx, y_part[i], x_part[i]] = -np.exp(-(t_ref - ts_part[i]) / tau)

            ## none-polarity Timesurface
            # sae[y[i], x[i]] = np.exp(-(t_ref-ts[i]) / tau)

        # fig = plt.figure()
        # fig.suptitle('Time surface')
        # plt.imshow(sae[time_idx], cmap='gray')
        # plt.xlabel("x [pixels]")
        # plt.ylabel("y [pixels]")
        # plt.colorbar()
        # # plt.savefig('time_surface.jpg')
        # plt.show()
    return sae


if __name__ == '__main__':
    file_name = "/home/Event_camera_action/DHP19/h5_dataset_7500_events/346x260/S10_session1_mov6_7500events.mat"
    whole_events = io.loadmat(file_name)['events'].astype(np.float32)

    # Important for DHP19
    # choose the camera_id for trainging and testing
    events = whole_events[whole_events[:, -1] == 0][:, :-1]

    # normalize the timestamps
    _min = events[:, 2].min()
    _max = events[:, 2].max()
    events[:, 2] = (events[:, 2] - _min) / (_max - _min)

    # change the original (x.y) ([1,346],[1,260]) to ([0,345],[0,259])
    events[:, 0] = events[:, 0] - 1
    events[:, 1] = events[:, 1] - 1

    # randomly choose part of the events, avoiding too large events for OOM
    row_total = events.shape[0]
    row_needed = int(1.0 * row_total)
    row_needed = min(row_needed, 1000000)
    row_sequence = np.random.choice(row_total, row_needed, replace=False, p=None)
    row_sequence.sort()
    events = events[row_sequence, :]

    start_time = time.time()
    # img = get_timeSurface(events[:,2], events[:,0].astype(np.int32), events[:,1].astype(np.int32), events[:,3],
    #                 repr_size=(260, 346), time_num=9)
    img = get_eventFrame(events[:, 2], events[:, 0].astype(np.int32), events[:, 1].astype(np.int32), events[:, 3],
                          repr_size=(260, 346), time_num=9)
    # img = get_eventCount(events[:, 2], events[:, 0].astype(np.int32), events[:, 1].astype(np.int32), events[:, 3],
    #                      repr_size=(260, 346), time_num=9)
    # img = get_eventAccuFrame(events[:, 2], events[:, 0].astype(np.int32), events[:, 1].astype(np.int32), events[:, 3],
    #                      repr_size=(260, 346), time_num=9)
    elapsed_time = time.time() - start_time
    print(f"Function execution time: {elapsed_time:.4f} seconds")

    # save_dir = '../vis/event_accu_frame/'
    # for time_idx in range(len(img)):
    #     fig = plt.figure()
    #     fig.suptitle('event_accu_frame')
    #     plt.imshow(img[time_idx], cmap='gray')
    #     plt.xlabel("x [pixels]")
    #     plt.ylabel("y [pixels]")
    #     plt.colorbar()
    #     plt.savefig(save_dir + str(time_idx) + '.jpg')
    # print("Finish.")