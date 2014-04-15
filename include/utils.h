
#ifndef __UTILS_H_INCLUDED__
#define __UTILS_H_INCLUDED__

__device__ void max_func(short *data, short *temp, short *out);

__device__ void min_func(short *data, short *temp, short *out);

__device__ void block_scan(int *data);

__device__ void radix_sort(int *data, int *temp1, int *temp2);

__global__ void test_kernel(int *test_int_data, short *test_short_data);

#endif
