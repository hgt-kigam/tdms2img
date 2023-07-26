import os
import time
import glob
import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from mpi4py import MPI


def tdms_to_npy(input_files, sample_num, input_count, patch_size, trace_start_num):
    sample_num_10deci = int(sample_num/10)
    data_temp=np.zeros(shape=(sample_num, patch_size))
    data=np.zeros(shape=(sample_num_10deci*input_count, patch_size))
    a = 0
    for input_file in input_files:
        tdms_file=TdmsFile(input_file)
        trace = tdms_file.groups()[0].channels()
        for i in range(patch_size):
            data_temp[:,i] = trace[trace_start_num+i]
        data[a:a+sample_num_10deci,:]=data_temp[::10,:]
        a += sample_num_10deci
    np.save(f'./data_{input_files[0][-22:-9]}_{input_files[-1][-22:-9]}.npy', data)
    return

def npy_to_image(input_files, desired_pixel, dpi, height):
    plt.figure(figsize=(desired_pixel/dpi, desired_pixel/dpi), dpi=dpi)
    for input_file in input_files:
        data = np.load(input_file)
        path = './image/'+f'{input_file[-36:-4]}'
        os.mkdir(path)
        for i in range(int(84000/height)):
            a = i * height
            plt.axis('off')
            plt.imshow(data[a:a+height,:], aspect="1", cmap='gray')
            plt.savefig(path+f'/{i+1}.png', bbox_inches='tight', pad_inches=0)
            plt.clf()
    return

comm = MPI.COMM_WORLD
time_begin = time.time()
rank = comm.Get_rank()
size = comm.Get_size()

input_files = sorted(glob.glob('*.tdms'))
length_data = len(input_files)
part_size = length_data//size

i_start = rank*part_size
i_end = i_start + part_size

input_files = input_files[i_start:i_end]

if(rank == (size-1)):
    i_end = length_data
# print(world_size)
patch_size = 224
trace_start_num = 300
sample_num = 30000
sample_num_10deci = 3000
tdms_to_npy(input_files, sample_num, length_data, patch_size, trace_start_num)

"""patch_size = 224
trace_start_num = 300
sample_num = 30000
sample_num_10deci = 3000
input_files = sorted(glob.glob('*.tdms'))
#input_files = input_files[:28*]
input_count = len(input_files)
tdms_to_npy(input_files, sample_num, input_count, patch_size, trace_start_num)

input_files = sorted(glob.glob('*.npy'))
desired_pixel = 291
dpi = 10
height = 224

npy_to_image(input_files, desired_pixel, dpi, height)"""




"""comm = MPI.COMM_WORLD
time_begin = time.time()

myrank = comm.Get_rank()
world_size = comm.Get_size()

length_data = 100

part_size = length_data//world_size

i_start = myrank*part_size
i_end = i_start + part_size

if(myrank == (world_size-1)):
    i_end = length_data

x = []
for i in range(i_start,i_end):
    x.append(i)

gathered_x = comm.gather(x, root=0)

if(myrank == 0):
    print(gathered_x)

time_end = time.time()
print("Total time taken = ",(time_end - time_begin))"""

	


