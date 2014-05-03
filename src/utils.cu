
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

//Adapted from Nvidia
__device__ void sum_reduction(int *data, int *out) {
    unsigned int id = threadIdx.x;
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
	if (id < s) {
	    data[id] += data[id + s];
	}
	__syncthreads();
    }
    if (id == 0) out[0] = data[0];
    __syncthreads();
}

/*
__device__ void reduce_sum(float *data, float *temp, float* out) {
    unsigned int tid = threadIdx.x;
    temp[tid] = data[tid];
    __syncthreads();
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
	if (tid < s) {
	    sdata[tid] += sdata[tid + s];
	}
	__syncthreads();
    }
    if (tid == 0) {
	out[0] = temp[0];
    }
}
*/

__device__ void max_func(short *data, short *temp, short *out) {
    unsigned int tid = threadIdx.x;
    temp[tid] = data[tid];
    __syncthreads();
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
	if (tid < s) {
	    temp[tid] = ((temp[tid+s] > temp[tid]) ? temp[tid+s] : temp[tid]);
	}
	__syncthreads();
    }
    if (tid == 0) {
	out[0] = temp[0];
    }
}

__device__ void max_func_special(short *data, short *temp, short *out, int stride, int offset) {
    unsigned int tid = threadIdx.x;
    temp[tid] = data[tid];
    __syncthreads();
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
	if ((tid-offset) < s && (tid-offset) > 0 && (tid-offset) % stride == 0) {
	    temp[tid] = ((temp[tid+s] > temp[tid]) ? temp[tid+s] : temp[tid]);
	}
	__syncthreads();
    }
    if (tid == 0) {
	out[0] = temp[0];
    }
}

__device__ void min_func(short *data, short *temp, short *out) {
    unsigned int tid = threadIdx.x;
    temp[tid] = data[tid];
    __syncthreads();
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
	if (tid < s) {
	    temp[tid] = ((temp[tid+s] < temp[tid]) ? temp[tid+s] : temp[tid]);
	}
	__syncthreads();
    }
    if (tid == 0) {
	out[0] = temp[0];
    }
}

__device__ void block_scan(int *data) {
	unsigned int tid = threadIdx.x;
	for (unsigned int d = 1; d<blockDim.x; d<<=1) {
		if ((tid + 1) % (d<<1) == 0) {
			data[tid] = data[tid] + data[tid - d];
		}
		__syncthreads();
	}

	if (tid==blockDim.x-1) {
		data[tid] = 0;
	}
	__syncthreads();
	int tmp;
	for (unsigned int d = blockDim.x>>1; d >= 1; d>>=1) {
		if ((tid + 1) % (d<<1) == 0) {
			tmp = data[tid - d];
			data[tid - d] = data[tid];
			data[tid] = tmp + data[tid];
		}
		__syncthreads();
	}
}

__device__ void block_scan(float *data) {
	unsigned int tid = threadIdx.x;
	for (unsigned int d = 1; d<blockDim.x; d<<=1) {
		if ((tid + 1) % (d<<1) == 0) {
			data[tid] = data[tid] + data[tid - d];
		}
		__syncthreads();
	}

	if (tid==blockDim.x-1) {
		data[tid] = 0;
	}
	__syncthreads();
	float tmp;
	for (unsigned int d = blockDim.x>>1; d >= 1; d>>=1) {
		if ((tid + 1) % (d<<1) == 0) {
			tmp = data[tid - d];
			data[tid - d] = data[tid];
			data[tid] = tmp + data[tid];
		}
		__syncthreads();
	}
}

__device__ void radix_sort(int *data, int *temp1, int *temp2) {
    unsigned int tid = threadIdx.x;
    unsigned int total = 0;
    unsigned int b = 0;
    for (unsigned int k=0; k<sizeof(int)*8; ++k) {
	b = (data[tid] & (1 << k)) == 0; //Actually opposite of bit
	temp1[tid] = b;
	temp2[tid] = b;
	__syncthreads();
	block_scan(temp1);
	total = temp1[blockDim.x-1] + temp2[blockDim.x-1];
	temp2[tid] = tid - temp1[tid] + total;
	temp1[tid] = b ? temp1[tid] : temp2[tid]; //Inverse of nvidia radix, account for b being !bit
	int tmp = data[tid];
	__syncthreads();
	data[temp1[tid]] = tmp;
	__syncthreads();
    }
}

__device__ void radix_sort_by_key(int *keys, int *data, int *temp1, int *temp2) {
    unsigned int tid = threadIdx.x;
    unsigned int total = 0;
    unsigned int b = 0;
    for (unsigned int k=0; k<sizeof(int)*8; ++k) {
	b = (keys[tid] & (1 << k)) == 0; //Actually opposite of bit
	temp1[tid] = b;
	temp2[tid] = b;
	__syncthreads();
	block_scan(temp1);
	total = temp1[blockDim.x-1] + temp2[blockDim.x-1];
	temp2[tid] = tid - temp1[tid] + total;
	temp1[tid] = b ? temp1[tid] : temp2[tid]; //Inverse of nvidia radix, account for b being !bit
	int tmp_data = data[tid];
	int tmp_key = keys[tid];
	__syncthreads();
	data[temp1[tid]] = tmp_data;
	keys[temp1[tid]] = tmp_key;
	__syncthreads();
    }
}

__global__ void test_kernel(int *test_int_data, short *test_short_data) {
    //Need blockdim of 256, one block
    __shared__ int test_int[1024];
    __shared__ int temp_int_1[1024];
    __shared__ int temp_int_2[1024];
    __shared__ short test_short[1024];
    __shared__ short temp_short[1024];
    __shared__ short out[1];
    unsigned int tid = threadIdx.x;
    test_int[tid] = 1024 - tid;
    temp_int_1[tid] = 1024 - tid;
    test_short[tid] = 1024 - tid;
    __syncthreads();

    if (tid == 0) printf("Running test kernel\n");
    max_func(test_short, temp_short, out);
    if (tid == 0) printf("Max: %d\n", out[0]);
    min_func(test_short, temp_short, out);
    if (tid == 0) printf("Min: %d\n", out[0]);
    block_scan(temp_int_1);
    if (tid == 0) {
	printf("Block scan: [");
	for (unsigned int i=0; i<1024; ++i) {
	    printf("%d, ", temp_int_1[i]);
	}
	printf("]\n");
    }
    __syncthreads();
    temp_int_1[tid] = 1024 - tid;
    block_scan(temp_int_1);
    if (tid == 0) {
	printf("Block scan: [");
	for (unsigned int i=0; i<1024; ++i) {
	    printf("%d, ", temp_int_1[i]);
	}
	printf("]\n");
    }
    __syncthreads();
    radix_sort(test_int, temp_int_1, temp_int_2);
    if (tid == 0) {
	printf("Radix Sort: [");
	for (unsigned int i=0; i<1024; ++i) {
	    printf("%d, ", test_int[i]);
	}
	printf("]\n");
    }
}

