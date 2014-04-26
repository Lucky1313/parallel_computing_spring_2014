


#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include "layout.h"
#include "utils.h"

using namespace std;

#define TILE_WIDTH 16
#define POP_SIZE 64
#define NUM_GEN 200

#define MIGRATION_FREQ 0
#define DISTANCE_WEIGHT 1
#define ANGLE_WEIGHT 100
#define OVERLAP_WEIGHT 1000


//Data is as follows:
//0 - size of each copy of data
//Node data (all are pairs of x, y):
//PWR
//GND
//INPUTS
//OUTPUTS
//TERMINALS - Always D-G-S
__constant__ short node_layout[32000];

//Probably a problem with pointers
__device__ int rand_int(curandState *state, int range) {
    return (int) (curand_uniform(state) * range);
}

__global__ void rand_setup_kernel(curandState *state) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int seed = (unsigned int) clock64();
    curand_init(seed + id, id, 0, &state[id]);
}

__global__ void ga_populate_kernel(curandState *state, short *pop_mem) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    curandState local_state = state[id];

    int mem_offset = id * node_layout[0];

    //Power and ground
    pop_mem[mem_offset] = rand_int(&local_state, node_layout[0] / 2);
    pop_mem[mem_offset+1] = 0;
    pop_mem[mem_offset+2] = rand_int(&local_state, node_layout[0] / 2);
    pop_mem[mem_offset+3] = node_layout[0] / 2;

    int off = 4;
    for (unsigned int i=0; i<node_layout[2]; ++i) {
	pop_mem[mem_offset+2*i+off] = 0;
	pop_mem[mem_offset+2*i+off+1] = rand_int(&local_state, node_layout[0] / 2);
    }
    off += node_layout[2] * 2;
    for (unsigned int i=0; i<node_layout[3]; ++i) {
	pop_mem[mem_offset+2*i+off] = node_layout[0] / 2;
	pop_mem[mem_offset+2*i+off+1] = rand_int(&local_state, node_layout[0] / 2);
    }
    off += node_layout[3] * 2;
    for (unsigned int i=off; i<node_layout[0]; i+=2) {
	if ((i - off) % 6 == 0) {
	    pop_mem[mem_offset+i] = rand_int(&local_state, node_layout[0] / 2);
	    
	    pop_mem[mem_offset+i+1] = rand_int(&local_state, node_layout[0] / 2);
	}
	else if ((i - off) % 6 == 2) {
	    pop_mem[mem_offset+i] = pop_mem[mem_offset+i-2] - 1;
	    pop_mem[mem_offset+i+1] = pop_mem[mem_offset+i-1] + 1;
	}
	else {
	    pop_mem[mem_offset+i] = pop_mem[mem_offset+i-4];
	    pop_mem[mem_offset+i+1] = pop_mem[mem_offset+i-3] + 2;
	}
    }
    state[id] = local_state;
}

__device__ float single_thread_fitness_func_mem(short* pop_mem, int mem_offset, int id) {
    //Distance between connected terminals
    int offset = node_layout[5];
    int node_offset = 1 + offset + node_layout[offset];
    int dist = 0;
    int size = 0;
    int angles = 0;
    short x[32]; //Will cause errors if any node has more that 32 terminals in one node
    short y[32]; //Needed because cuda does not allow dynamically sized arrays
    short t, xdiff, ydiff;
    for (short i=0; i<node_layout[offset]; ++i) {
        size = node_layout[offset+i+1];
	for (short j=0; j<size; ++j) {
	    t = node_layout[node_offset];
	    x[j] = pop_mem[mem_offset+t];
	    y[j] = pop_mem[mem_offset+t+1];
	    for (short k=0; k<j; ++k) {
		xdiff = x[j] - x[k];
		ydiff = y[j] - y[k];
		dist += xdiff * xdiff + ydiff * ydiff;

		//Angles
		angles += (xdiff != 0) + (ydiff != 0) - 1;
	    }
	    ++node_offset;
	}
    }
    angles = angles * ANGLE_WEIGHT;
    dist = dist * DISTANCE_WEIGHT;

    int overlap = 0;
    int off = mem_offset + (node_layout[2] + node_layout[3]) * 2 + 4;
    short x1, y1, x2, y2;
    for (unsigned int i=0; i<node_layout[4]; ++i) {
	for (unsigned int j=i+1; j<node_layout[4]; ++j) {
	    x1 = pop_mem[off+i*3];
	    y1 = pop_mem[off+i*3+1];
	    x2 = pop_mem[off+j*3];
	    y2 = pop_mem[off+j*3+1];
	    if ((x1 - x2) < 2 && (x1 - x2) > -2 &&
		(y1 - y2) < 3 && (y1 - y2) > -3) {
		++overlap;
	    }
	}
    }
    overlap = overlap * OVERLAP_WEIGHT;
    
    int value = dist + angles + overlap;
    //Print some useful info
    if (id == 0)
    {
	printf("Thread 0\n");
	printf("Layout dump: [");
	for (int i=0; i<node_layout[0]/2; ++i) {
	    printf("(%d, %d)", pop_mem[mem_offset+i*2], pop_mem[mem_offset+i*2+1]);
	}
	printf("]\n");
	printf("Fitness function for thread %d: %d (%d, %d, %d)\n", id, value, dist, angles, overlap);
    }
    return value;
}

__device__ void crossover() {
    int pos;
    int tid = threadIdx.x;
    for (unsigned int i=0; i<TILE_WIDTH; ++i) {
	if (scores_id[i] == tid) {
	    pos = i;
	}
    }
    char half = pos % 2;
    int complement_pos = (half == 0 ? pos + 1 : pos - 1);
    int complement_tid = scores_id[complement_pos];

    curandState local_state = state[id];
    int num_transistors = node_layout[4]; 
    temp2[tid] = rand_int(&local_state, num_transistors / 2) + (half ? num_transistors / 2 : 0);
    __syncthreads();
    int width = (temp2[tid] - temp2[complement_tid]) * (half ? 6 : -6) + 6;
    int start_offset = (half ? temp2[complement_tid] : temp2[tid]) * 6 + (node_layout[2] + node_layout[3]) * 2 + 4;
    int offset = mem_offset + start_offset;
    for (unsigned int i=0; i<width; ++i) {
	temp_mem[mem_offset+i] = pop_mem[offset+i];
    }
    __syncthreads();
    int complement_offset = (blockIdx.x*blockDim.x+complement_tid) * node_layout[0] + start_offset;
    for (unsigned int i=0; i<width; ++i) {
	pop_mem[complement_offset+i] = temp_mem[mem_offset+i];
    }
    __syncthreads();
}

__device__ void mutation() {

}

__global__ void ga_kernel(curandState *state, short *pop_mem, short *temp_mem, int *fit_mem) {
    __shared__ int fit_scores[TILE_WIDTH];
    __shared__ int scores_id[TILE_WIDTH];
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    int mem_offset = id * node_layout[0];
    
    fit_scores[threadIdx.x] = single_thread_fitness_func_mem(pop_mem, mem_offset, id);
    scores_id[threadIdx.x] = threadIdx.x;

    __shared__ int temp1[TILE_WIDTH];
    __shared__ int temp2[TILE_WIDTH];
    radix_sort_by_key(fit_scores, scores_id, temp1, temp2);

    if (id == 0) {
    	printf("Fitness_func: [");
    	for (unsigned int i=0; i<TILE_WIDTH; ++i) {
    	    printf("%d, ", fit_scores[i]);
    	}
    	printf("]\nPositions: [");
	for (unsigned int i=0; i<TILE_WIDTH; ++i) {
	    printf("%d, ", scores_id[i]);
	}
	printf("]\n");
    }

    //Selection
    
    //Mutation
    //Number of mutations is POP/32
    int num_mut = POP_SIZE>>5;
    if (tid % num_mut == 0) {
	
    }
    

    state[id] = local_state;
}

__global__ void ga_migration_kernel(short *pop_mem) {

}

__global__ void ga_print_kernel(short *pop_mem) {
    //int id = blockIdx.x*blockDim.x + threadIdx.x;
    //int mem_offset = id * node_layout[0];

    
}

int term_pos(Terminal *term, int offset_data, int offset_in, int offset_out) {
    switch(term->type) {
    case 'D':
	return 2 + offset_in + offset_out + (term->num - 1) * 3; //Order D-G-S
    case 'G':
	return 3 + offset_in + offset_out + (term->num - 1) * 3;
    case 'S':
	return 4 + offset_in + offset_out + (term->num - 1) * 3;
    case 'P':
	return 0; //Always first
    case 'Z':
	return 1; //Always second
    case 'I':
	return 1 + term->num; //Directly follows ground, term num starts at 1, so only offset 1
    case 'O':
	return 1 + offset_in + term->num; // Follows inputs
    default:
	cout << "Improper terminal" << endl;
	return -1;
    }
}

void create_node_data_array(Layout *main_layout, short *node_data, int offset_data, int offset_in, int offset_out) {
    Node* node;
    node_data[offset_data] = main_layout->nodes_size();
    int node_offset = 1 + offset_data + main_layout->nodes_size();
    for (unsigned int i=0; i<main_layout->nodes_size(); ++i) {
	node = main_layout->get_node(i);
	node_data[i+1+offset_data] = node->terms.size();
	for (unsigned int j=0; j<node->terms.size(); ++j) {
	    node_data[node_offset] = term_pos(node->terms[j], offset_data, offset_in, offset_out);
	    ++node_offset;
	}
    }
}

void launch_ga(Layout *main_layout) {
    //Specify block size
    const dim3 block_size(TILE_WIDTH);
    //Assume POP_SIZE is multiple of block size
    const dim3 num_blocks(POP_SIZE/block_size.x);

    int num_terminals = (main_layout->trans_size()) * 3;
    int offset_in = main_layout->in_size();
    int offset_out = main_layout->out_size();
    int trans_offset = offset_in + offset_out + 2;
    short per_copy_size = (trans_offset + num_terminals) * 2 * sizeof(short);
    int thread_count = num_blocks.x * num_blocks.y * block_size.x * block_size.y;
    int total_mem = thread_count * per_copy_size;

    cout << "Read in Layout. " << endl;
    cout << "Offset: " << trans_offset << endl;
    cout << "Number of transistor terminals: " << num_terminals << endl;


    cout << "\n\nBlock Allocation:\nDefined Tile size: " << TILE_WIDTH << endl;
    cout << "Defined population size: " << POP_SIZE << endl;
    cout << "Number of blocks: " << num_blocks.x * num_blocks.y << endl;
    cout << "Number of threads per block: " << block_size.x * block_size.y<< endl;
    cout << "Total number of threads: " << thread_count << endl;

    //Allocate memory
    cout << "\n\nMemory:\nSize of short: " << sizeof(short) << " bytes" << endl;
    cout << "Size of layout copy: " << per_copy_size << " bytes" << endl;
    cout << "Total layout memory use: " << total_mem << " bytes" << endl;

    int size = 0;
    Node* node;
    for (unsigned int i=0; i<main_layout->nodes_size(); ++i) {
	node = main_layout->get_node(i);
	size += node->terms.size();
    }

    int offset_data = 6;
    int node_mem_size = 1 + main_layout->nodes_size() + size + offset_data;
    short node_data[node_mem_size];

    node_data[0] = (trans_offset + num_terminals) * 2;
    node_data[1] = per_copy_size;
    node_data[2] = offset_in;
    node_data[3] = offset_out;
    node_data[4] = num_terminals / 3;
    node_data[5] = offset_data;

    cout << "Node size: " << node_mem_size << " numbers" << endl;
    cout << "Node memory use: " << node_mem_size * sizeof(short) << " bytes\n" << endl;
    
    create_node_data_array(main_layout, node_data, offset_data, offset_in, offset_out);
    cout << "Node data: [" << node_data[0] << ", " << node_data[1] << ", " << node_data[2];
    cout << ", " << node_data[3] << ", " << node_data[4] << ", " << node_data[5] << "]\n" << endl;
    
    //Random number generation setup
    curandState *rand_states = 0;
    cudaMalloc((void **)&rand_states, thread_count * sizeof(curandState));
    rand_setup_kernel<<<num_blocks, block_size>>>(rand_states);

    short *pop_mem = 0;
    short *temp_mem = 0;
    //short *node_mem = 0;
    int *fit_mem = 0;
    short *run_num = 0;

    cudaMalloc((void**)&pop_mem, total_mem);
    cudaMalloc((void**)&temp_mem, total_mem);
    //cudaMalloc((void**)&node_mem, node_mem_size * sizeof(short));
    cudaMalloc((void**)&fit_mem, thread_count * sizeof(int));
    cudaMalloc((void**)&run_num, sizeof(short));

    //cudaMemcpy(node_mem, node_data, node_mem_size * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(node_layout, node_data, node_mem_size * sizeof(short));
    cudaMemset(run_num, 0, sizeof(short));

    ga_populate_kernel<<<num_blocks, block_size>>>(rand_states, pop_mem);

    for (int i=0; i<NUM_GEN; ++i) {
	ga_kernel<<<num_blocks, block_size>>>(rand_states, pop_mem, temp_mem, fit_mem);
	//ga_print_kernel<<<num_blocks, block_size>>>(pop_mem);
	if (i % 10 == 9) {
	    ga_migration_kernel<<<num_blocks, block_size>>>(pop_mem);
	}
    }
    short *host_pop = 0;
    host_pop = (short*)malloc(total_mem);

    cudaMemcpy(host_pop, pop_mem, total_mem, cudaMemcpyDeviceToHost);
/*
    cout << "Util test kernel" << endl;
    int test_int[64] = {38, 56, 41, 0, 43, 51, 18, 45, 34, 63, 37, 54, 1, 59, 32, 28, 40, 42, 17, 7, 22, 25, 8, 36, 4, 12, 23, 35, 29, 44, 52, 31, 26, 16, 15, 14, 33, 48, 5, 53, 2, 11, 24, 62, 20, 30, 39, 27, 3, 61, 47, 6, 19, 50, 55, 13, 58, 46, 9, 49, 21, 10, 60, 57};
    short test_short[64] = {38, 56, 41, 0, 43, 51, 18, 45, 34, 63, 37, 54, 1, 59, 32, 28, 40, 42, 17, 7, 22, 25, 8, 36, 4, 12, 23, 35, 29, 44, 52, 31, 26, 16, 15, 14, 33, 48, 5, 53, 2, 11, 24, 62, 20, 30, 39, 27, 3, 61, 47, 6, 19, 50, 55, 13, 58, 46, 9, 49, 21, 10, 60, 57};

    int *test_int_data = 0;
    short *test_short_data = 0;
    cudaMalloc((void**)&test_int_data, 64*sizeof(int));
    cudaMalloc((void**)&test_short_data, 64*sizeof(short));

    cudaMemcpy(test_int_data, test_int, 64*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(test_short_data, test_short, 64*sizeof(short), cudaMemcpyHostToDevice);
    
    test_kernel<<<1, 64>>>(test_int_data, test_short_data);
*/
    free(host_pop);
    
    cudaFree(pop_mem);
    cudaFree(fit_mem);
    cudaFree(run_num);
}

