import os
import time
import glob
import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from mpi4py import MPI

start = time.time()

comm = MPI.COMM_WORLD
time_begin = time.time()
rank = comm.Get_rank()
size = comm.Get_size()

input_files = sorted(glob.glob('/datadisk2/Janggi_Data/2019.07.17-08.14/IDAS DATA/190731-190806/*.tdms'))
length_data = len(input_files)
part_size = length_data//size

i_start = rank*part_size
i_end = i_start + part_size

input_files = input_files[i_start:i_end]


#print(len(input_files), i_start, i_end)

if(rank == (size-1)):
    i_end = length_data

def tdms_to_npy(input_files, sample_num, input_count, patch_size, trace_start_num):
    sample_num_10deci = int(sample_num/10)
    data_temp=np.zeros(shape=(sample_num, patch_size))
    data=np.zeros(shape=(sample_num_10deci*28, patch_size))
    a = 0
    for i in range(int(input_count / 28)):
        b = i*28
        input_files_temp = input_files[b:b+28]
        time = TdmsFile.read_metadata(input_files_temp[0]).properties['GPSTimeStamp']
        a = 0
        for input_file in input_files_temp:
            tdms_file=TdmsFile(input_file)
            trace = tdms_file.groups()[0].channels()
            for i in range(patch_size):
                data_temp[:,i] = trace[trace_start_num+i]
            data[a:a+sample_num_10deci,:]=data_temp[::10,:]
            a += sample_num_10deci
        np.savez(f'./npy_mpi/data_{input_files_temp[0][-22:-9]}_{input_files_temp[-1][-22:-9]}.npy', time=time, data=data)
    return

patch_size = 224
trace_start_num = 300
sample_num = 30000


input_count = len(input_files)
if input_count % 28 == 0:
    tdms_to_npy(input_files, sample_num, input_count, patch_size, trace_start_num)
else:
    print("check input files")
time_end = time.time()
print("Total elapsed time",time.time()-start)
