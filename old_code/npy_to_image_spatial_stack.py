import os
import time
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

start = time.time()

comm = MPI.COMM_WORLD
time_begin = time.time()
rank = comm.Get_rank()
size = comm.Get_size()

input_files = sorted(glob.glob('./npz/*.npz'))
length_data = len(input_files)
part_size = length_data//size

i_start = rank*part_size
i_end = i_start + part_size

input_files = input_files[i_start:i_end]

if(rank == (size-1)):
    i_end = length_data


def np2datetime(np_datetime):
    np_datetime = np_datetime.astype(datetime.datetime)
    strf_datetime = np_datetime.strftime('%y%m%d_%H%M%S_%f')[:-3]
    return strf_datetime


def npy_to_image(input_files, height):
    b = 0
    for input_file in input_files:
        with np.load(input_file) as npz:
            time = npz['time']
            data = npz['data']
        # path = './image/'+f'{input_file[-36:-4]}/'
        # os.mkdir(path)
        for i in range(int(84000/height)):
            a = i * height
            time_delta = np.timedelta64(10*a, 'ms')
            name = np2datetime(time+time_delta)
            plt.axis('off')
            plt.imshow(data[a:a+height,:], aspect="1", cmap='gray')
            plt.savefig('./image_96'+name+'.png', format='png', bbox_inches='tight', pad_inches=0)
            plt.clf()
        b += 1
        print(b/len(input_files)*100,"%  complite")
    return
desired_pixel = 125
dpi = 10
plt.figure(figsize=(desired_pixel/dpi, desired_pixel/dpi), dpi=dpi)
height = 96

npy_to_image(input_files, height)

print("Total elapsed time",time.time()-start)