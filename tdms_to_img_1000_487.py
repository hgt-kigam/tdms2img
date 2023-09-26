import time
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from nptdms import TdmsFile
from pkprocess import perc_clip

start = time.time()

comm = MPI.COMM_WORLD
time_begin = time.time()
rank = comm.Get_rank()
size = comm.Get_size()

input_files = sorted(glob.glob('/datadisk2/Janggi_Data/2019.07.17-08.14/IDAS DATA/190717-190723/*.tdms'))
length_data = len(input_files)
part_size = length_data//size

i_start = rank*part_size
i_end = i_start + part_size

input_files = input_files[i_start:i_end]

if (rank == (size-1)):
    i_end = length_data


def tdms_to_img(input_files, start_channel_num, end_channel_num, rank):
    b = 0
    for input_file in input_files:
        time = TdmsFile.read_metadata(input_file).properties['GPSTimeStamp']
        tdms_file = TdmsFile(input_file)
        trace = tdms_file.groups()[0].channels()
        data = np.array(trace)[start_channel_num:end_channel_num, :]
        # data = tdms_file.groups()[0].channels()[start_channel_num:end_channel_num, :]
        data = perc_clip(data, CLIP)
        for i in range(int(sample_num/height)):
            a = i * height
            time_delta = np.timedelta64(a, 'ms')
            name = np2datetime(time+time_delta)
            image = data[:, a:a+height].T
            plt.imshow(image, cmap='gray_r')
            plt.axis('off')
            plt.savefig(f'./image_1000_487/{name}.png', format='png', bbox_inches='tight', pad_inches=0)
            plt.clf()
        b += 1
        print(f'{rank}:', b/len(input_files)*100, "%d complite")
    return


def np2datetime(np_datetime):
    np_datetime = np_datetime.astype(datetime.datetime)
    strf_datetime = np_datetime.strftime('%y%m%d_%H%M%S_%f')[:-3]
    return strf_datetime


desired_pixel = 1299
dpi = 100
plt.figure(figsize=(desired_pixel/dpi, desired_pixel/dpi), dpi=dpi)

sample_num = 30000
height = 1000
CLIP = 90
start_channel_num = 162
end_channel_num = 649

tdms_to_img(input_files, start_channel_num, end_channel_num, rank)
print("Total elapsed time", time.time()-start)
