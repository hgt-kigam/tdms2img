import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

start = time.time()

comm = MPI.COMM_WORLD
time_begin = time.time()
rank = comm.Get_rank()
size = comm.Get_size()

input_files = sorted(glob.glob('./npy_mpi/*.npy'))
length_data = len(input_files)
part_size = length_data//size

i_start = rank*part_size
i_end = i_start + part_size

input_files = input_files[i_start:i_end]

if(rank == (size-1)):
    i_end = length_data

def npy_to_image(input_files, height):
    for input_file in input_files:
        data = np.load(input_file)
        path = './image_mpi/'+f'{input_file[-36:-4]}'
        os.mkdir(path)
        for i in range(int(84000/height)):
            a = i * height
            plt.axis('off')
            plt.imshow(data[a:a+height,:], aspect="1", cmap='gray')
            plt.savefig(path+'/{i+1}.png', bbox_inches='tight', pad_inches=0)
            plt.clf()
    return
desired_pixel = 291
dpi = 10
plt.figure(figsize=(desired_pixel/dpi, desired_pixel/dpi), dpi=dpi)
height = 224

npy_to_image(input_files, height)

print("Total elapsed time",time.time()-start)