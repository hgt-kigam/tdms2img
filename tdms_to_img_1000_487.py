import os
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


def tdms_to_img(input_files, start_channel_num, end_channel_num, height, clip, rank):
    b = 0
    for input_file in input_files:
        time = TdmsFile.read_metadata(input_file).properties['GPSTimeStamp']
        tdms_file = TdmsFile(input_file)
        trace = tdms_file.groups()[0].channels()
        sample_num = trace.shape[1]
        data = np.array(trace)[start_channel_num:end_channel_num, :]
        if clip == 100:
            output_path = './image_1000_487/'
            os.mkdir(output_path)
        else:
            data = perc_clip(data, clip)
            output_path = './image_1000_487_clip/'
            os.mkdir(output_path)
        for i in range(int(sample_num/height)):
            a = i * height
            time_delta = np.timedelta64(a, 'ms')
            name = np_to_datetime(time+time_delta) + '.png'
            image = data[:, a:a+height].T
            plt.imshow(image, cmap='gray_r')
            plt.axis('off')
            plt.savefig(output_path + name, format='png', bbox_inches='tight', pad_inches=0)
            plt.clf()
        b += 1
        print(f"{rank}:", f"{ b/len(input_files)*100}% complite")
    return


def np_to_datetime(np_datetime):
    np_datetime = np_datetime.astype(datetime.datetime)
    strf_datetime = np_datetime.strftime('%y%m%d_%H%M%S_%f')[:-3]
    return strf_datetime


desired_pixel = 1299
dpi = 100
plt.figure(figsize=(desired_pixel/dpi, desired_pixel/dpi), dpi=dpi)

height = 1000
clip = 90
start_channel_num = 162
end_channel_num = 649

tdms_to_img(input_files, start_channel_num, end_channel_num, height, clip, rank)
print("Total elapsed time", time.time()-start)
