


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

//Used for autotuning
#ifndef TILE_WIDTH
    #define TILE_WIDTH 1024
#endif
#ifndef POP_SIZE
    #define POP_SIZE (1 << 17)
#endif
#ifndef NUM_GEN
    #define NUM_GEN 5000
#endif

#define NUM_BLOCKS POP_SIZE / TILE_WIDTH
#define MUTATION 30

//Weights for fitness function
#define MIGRATION_FREQ 0
#define DISTANCE_WEIGHT 10
#define ANGLE_WEIGHT 10
#define OVERLAP_WEIGHT 1000
#define LEFT_WEIGHT 250
#define RIGHT_WEIGHT 250
#define UP_WEIGHT 250
#define DOWN_WEIGHT 250

//Data is as follows:
//0 - size of each copy of data
//Node data (all are pairs of x, y):
//PWR
//GND
//INPUTS
//OUTPUTS
//TERMINALS - Always D-G-S
__constant__ short node_layout[32000];

//Retrieve random integer in kernel
__device__ int rand_int(curandState *state, int range) {
    return (int) (curand_uniform(state) * range);
}

//Set up a unique random generator for each thread
__global__ void rand_setup_kernel(curandState *state) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int seed = (unsigned int) clock64();
    curand_init(seed + id, id, 0, &state[id]);
}

//Create random layout based off node data
__global__ void ga_populate_kernel(curandState *state, short *pop_mem) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    curandState local_state = state[id];

    int mem_offset = id * node_layout[0];

    //Power and ground
    pop_mem[mem_offset] = rand_int(&local_state, node_layout[0] / 2);
    pop_mem[mem_offset+1] = 0;
    pop_mem[mem_offset+2] = rand_int(&local_state, node_layout[0] / 2);
    pop_mem[mem_offset+3] = node_layout[0] / 2;

    //Inputs
    int off = 4;
    for (unsigned int i=0; i<node_layout[2]; ++i) {
	pop_mem[mem_offset+2*i+off] = 0;
	pop_mem[mem_offset+2*i+off+1] = rand_int(&local_state, node_layout[0] / 2);
    }
    //Outputs
    off += node_layout[2] * 2;
    for (unsigned int i=0; i<node_layout[3]; ++i) {
	pop_mem[mem_offset+2*i+off] = node_layout[0] / 2;
	pop_mem[mem_offset+2*i+off+1] = rand_int(&local_state, node_layout[0] / 2);
    }
    //Transistors
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
    //Update random state
    state[id] = local_state;
}

//Fitness function
__device__ float single_thread_fitness_func_mem(short* pop_mem, int mem_offset, int id) {
    //Distance between connected terminals
    int offset = node_layout[5];
    int node_offset = 1 + offset + node_layout[offset];
    int dist = 0;
    int size = 0;
    int angles = 0;
    short x[32]; //Will cause errors if any node has more that 32 terminals in one node
    short y[32]; //Needed because cuda does not allow dynamically sized arrays
    short max_x = 0;
    short max_y = 0;
    short t, xdiff, ydiff;
    //Calculates the distance between each terminal and all others in the same node
    //and the number of turns a wire connecting them will have to make
    for (short i=0; i<node_layout[offset]; ++i) {
        size = node_layout[offset+i+1];
	for (short j=0; j<size; ++j) {
	    t = node_layout[node_offset];
	    x[j] = pop_mem[mem_offset+t];
	    y[j] = pop_mem[mem_offset+t+1];
	    if (x[j] > max_x) max_x = x[j];
	    if (y[j] > max_y) max_y = y[j];
	    for (short k=0; k<j; ++k) {
		xdiff = x[j] - x[k];
		ydiff = y[j] - y[k];
                //Distance
		dist += xdiff * xdiff + ydiff * ydiff;

		//Angles
		angles += (xdiff != 0) + (ydiff != 0) - 1;
	    }
	    ++node_offset;
	}
    }
    //Weighting the calculated values
    angles = angles * ANGLE_WEIGHT;
    dist = dist * DISTANCE_WEIGHT;

    int overlap = 0;
    int off = mem_offset + (node_layout[2] + node_layout[3]) * 2 + 4;
    short x1, y1, x2, y2;
    //Calculates the number of transistors that overlap
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

    int left = 0;
    int right = 0;
    int up = 0;
    int down = 0;
    int pos = 0;
    //Inputs to left side
    for (short i=0; i<node_layout[2]; ++i) {
	pos = pop_mem[mem_offset+i+4];
  	left += pos * pos * LEFT_WEIGHT;
    }
    //Outputs to right side
    for (short i=0; i<node_layout[3]; ++i) {
  	pos = pop_mem[mem_offset+i+node_layout[2]*2+4] - max_x; 
  	right += pos * pos * RIGHT_WEIGHT;
    }
    //Power to top
    pos = pop_mem[mem_offset+1];
    up = pos * pos * UP_WEIGHT;
    //Ground to bottom
    pos = pop_mem[mem_offset+3] - max_y;
    down = pos * pos * DOWN_WEIGHT;
    
    int value = dist + angles + overlap + left + right + up + down;
    return value;
}

//Performs a crossover between two individuals, keeping the transistors and the power/ground/input/outputs separate
__device__ void crossover(curandState *state, short *pop_mem, short *temp_mem, int mem, int off1, int off2) {

    //IO Crossover
    int num_io = node_layout[2] + node_layout[3] + 2;
    int cross1 = rand_int(state, num_io);

    for (int i=0; i<cross1*2; ++i) {
	temp_mem[mem+i] = pop_mem[off1+i];
    }
    for (int i=cross1*2; i<num_io*2; ++i) {
	temp_mem[mem+i] = pop_mem[off2+i];
    }

    //Terminal Crossover
    int num_transistors = node_layout[4];
    int cross2 = rand_int(state, num_transistors);

    mem += num_io * 2;
    off1 += num_io * 2;
    off2 += num_io * 2;

    for (int i=0; i<cross2*6; ++i) {
	temp_mem[mem+i] = pop_mem[off1+i];
    }
    for (int i=cross2*6; i<num_transistors*6; ++i) {
	temp_mem[mem+i] = pop_mem[off2+i];
    }
}

//Performs a mutation with the chance being specified by the MUTATION global parameter, out of 100
__device__ void mutation(curandState *state, short *temp_mem, int mem_offset) {

    int chance = rand_int(state, 100);
    if (chance < MUTATION) {
	int pick = rand_int(state, node_layout[0]);
	int num_io = node_layout[2] + node_layout[3] + 2;
	if (pick < num_io * 2) {
	    temp_mem[mem_offset+pick] = rand_int(state, node_layout[0] / 2);
	}
	else {
	    int dgs = ((pick - num_io * 2) / 2) % 3;
	    int xy = pick % 2;
	    int rand = rand_int(state, node_layout[0] / 2);
	    //Modify the whole transistor to make sure they maintain the right positioning
	    switch(dgs) {
	    case 0:
		temp_mem[mem_offset+pick] = rand;
		temp_mem[mem_offset+pick+2] = rand + (xy ? 1 : -1);
		temp_mem[mem_offset+pick+4] = rand + (xy ? 2 : 0);
		break;
	    case 1:
		temp_mem[mem_offset+pick-2] = rand + (xy ? -1 : 1);
		temp_mem[mem_offset+pick] = rand;
		temp_mem[mem_offset+pick+2] = rand + 1;
		break;
	    case 2:
		temp_mem[mem_offset+pick-4] = rand + (xy ? -2 : 0);
		temp_mem[mem_offset+pick-2] = rand - 1;
		temp_mem[mem_offset+pick] = rand;
		break;
	    default:
		printf("Err\n");
	    }
	}
    }
    
}

//Simple in-kernel binary search, returning the node closest to but less than the 'point' value
__device__ int bin_search(float *normalized_data, float point, int begin, int end) {
    int pivot = (end - begin) / 2 + begin;
    if (point >= normalized_data[pivot]) {
	if (pivot == end - 1) return pivot;
	return bin_search(normalized_data, point, pivot, end);
    }
    else {
	if (pivot == begin) return begin;
	return bin_search(normalized_data, point, begin, pivot);
    }
}

//Genetic Algorithm kernel
__global__ void ga_kernel(curandState *state, short *pop_mem, short *temp_mem, short *best_mem, int *score_mem) {
    __shared__ int fit_scores[TILE_WIDTH];
    __shared__ int scores_id[TILE_WIDTH];
    __shared__ int temp1[TILE_WIDTH];
    __shared__ int temp2[TILE_WIDTH];
    __shared__ float normalized_fit_scores[TILE_WIDTH];
    __shared__ int sum;
    int tid = threadIdx.x;
    int id = blockIdx.x*blockDim.x + tid;
    int mem_offset = id * node_layout[0];

    //Obtain fitness scores for each thread/individual
    fit_scores[tid] = single_thread_fitness_func_mem(pop_mem, mem_offset, id);
    scores_id[tid] = tid;
    __syncthreads();

    //Sort fitness functions least to greatest
    radix_sort_by_key(fit_scores, scores_id, temp1, temp2);

    //Copy best layout to best memory if it is better than the previous best
    if (score_mem[blockIdx.x] > fit_scores[0] || score_mem[blockIdx.x] == 0) {
	int best_id = scores_id[0];
	if (tid == 0) {
	    //printf("Better score, overwriting %d with %d\n", score_mem[blockIdx.x], fit_scores[0]);
	    score_mem[blockIdx.x] = fit_scores[0];
	}
	int best_offset = blockIdx.x * node_layout[0];
	for (unsigned int i=tid; i<node_layout[0]; i += TILE_WIDTH) {
	    best_mem[best_offset+i] = pop_mem[best_id*node_layout[0]+i];
	    //if (blockIdx.x == 0) printf("%d, %d, %d, %d\n", tid, i, best_offset+i, best_id*node_layout[0]+i);
	}
    }
    /*
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
    */
    //Normalize and then scan the scores in order for roulette wheel choosing
    //Slight variation on: http://en.wikipedia.org/wiki/Selection_(genetic_algorithm)
    temp1[tid] = fit_scores[tid];
    if (tid == 0) sum = 0;
    __syncthreads();
    sum_reduction(temp1, &sum);
    normalized_fit_scores[tid] = (1 - ((float) fit_scores[tid] / (float) sum)) / (TILE_WIDTH - 1);
    __syncthreads();

    /*
    if (id == 0) {
	printf("Normalized scores: [");
	for (unsigned int i=0; i<TILE_WIDTH; ++i) {
	    printf("%f, ", normalized_fit_scores[i]);
	}
	printf("]\n");
    }
    */
    block_scan(normalized_fit_scores);

    /*
    if (id == 0) {
	printf("Summed Normalized scores: [");
	for (unsigned int i=0; i<TILE_WIDTH; ++i) {
	    printf("%f, ", normalized_fit_scores[i]);
	}
	printf("]\n");
    }
    */
    //Pick two individuals for crossover
    curandState local_state = state[id];
    
    float pick1 = curand_uniform(&local_state);
    float pick2 = curand_uniform(&local_state);

    int pos1 = bin_search(normalized_fit_scores, pick1, 0, TILE_WIDTH);
    int pos2 = bin_search(normalized_fit_scores, pick2, 0, TILE_WIDTH);

    int id1 = scores_id[pos1];
    int id2 = scores_id[pos2];

    int off1 = (id1 + blockDim.x*blockIdx.x) * node_layout[0];
    int off2 = (id2 + blockDim.x*blockIdx.x) * node_layout[0];

    //printf("New child from parents %d (%d) and %d (%d)\n", id1, fit_scores[pos1], id2, fit_scores[pos2]);
    //Crossover
    //Store new generation in temp_mem
    crossover(&local_state, pop_mem, temp_mem, mem_offset, off1, off2);
    
    //Mutation
    mutation(&local_state, temp_mem, mem_offset);

    //Return random state
    state[id] = local_state;
}

//Kernel to select the best layout of all blocks 
__global__ void champion_kernel(short *best_mem, int *score_mem) {
    //Copy to shared memory
    __shared__ int scores[NUM_BLOCKS];
    __shared__ int scores_id[NUM_BLOCKS];
    __shared__ int temp1[NUM_BLOCKS];
    __shared__ int temp2[NUM_BLOCKS];
    int tid = threadIdx.x;
    scores[tid] = score_mem[tid];
    scores_id[tid] = tid;
    __syncthreads();
    //Sort by fitness scores
    radix_sort_by_key(scores, scores_id, temp1, temp2);

    //Copy best score and layout to last position in array
    if (score_mem[NUM_BLOCKS] > scores[0] || score_mem[NUM_BLOCKS] == 0) {
	if (tid == 0) {
	    score_mem[NUM_BLOCKS] = scores[0];
	}
	int best_offset = scores_id[0] * node_layout[0];
	for (unsigned int i=tid; i<node_layout[0]; i += NUM_BLOCKS) {
	    //printf("Champion copying point %d\n", i);
	    //printf("Place %d, Copying from %d to %d\n", i, NUM_BLOCKS*node_layout[0]+i, best_offset+i);
	    best_mem[NUM_BLOCKS*node_layout[0]+i] = best_mem[best_offset+i];
	}
    }
}

//Used for generating the position in memory based on the component
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

//Create the array containing all the layout data based on the read file
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

//Genetic algorithm function launched from main file
void launch_ga(Layout *main_layout) {
    //Specify block size
    const dim3 block_size(TILE_WIDTH);
    //Assume POP_SIZE is multiple of block size
    const dim3 num_blocks(NUM_BLOCKS);

    //Output some useful data
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
    cout << "Number of blocks: " << NUM_BLOCKS << endl;
    cout << "Number of threads per block: " << block_size.x * block_size.y<< endl;
    cout << "Total number of threads: " << thread_count << endl;

    //Allocate memory
    cout << "\n\nMemory:\nSize of short: " << sizeof(short) << " bytes" << endl;
    cout << "Size of int: " << sizeof(int) << " bytes" << endl;
    cout << "Size of float: " << sizeof(float) << " bytes" << endl;
    cout << "Size of layout copy: " << per_copy_size << " bytes" << endl;
    cout << "Total layout memory use: " << total_mem << " bytes" << endl;

    cout << "Running for " << NUM_GEN*2 << " generations." << endl;

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
    short *best_mem = 0;
    int *score_mem = 0;

    //Memory allocation
    cudaMalloc((void**)&pop_mem, total_mem);
    cudaMalloc((void**)&temp_mem, total_mem);
    cudaMalloc((void**)&best_mem, (num_blocks.x+1) * per_copy_size);
    cudaMalloc((void**)&score_mem, (num_blocks.x+1) * sizeof(int));

    //Copy node data to global for use in kernel
    cudaMemcpyToSymbol(node_layout, node_data, node_mem_size * sizeof(short));

    cudaMemset(score_mem, 0, (num_blocks.x+1) * sizeof(int));

    ga_populate_kernel<<<num_blocks, block_size>>>(rand_states, pop_mem);

    //Call genetic algorithm twice, alternating storing the data in pop_mem and temp_mem
    for (int i=0; i<NUM_GEN; ++i) {
	ga_kernel<<<num_blocks, block_size>>>(rand_states, pop_mem, temp_mem, best_mem, score_mem);
	ga_kernel<<<num_blocks, block_size>>>(rand_states, temp_mem, pop_mem, best_mem, score_mem);
    }
    
    //printf("Calling champion kernel with: %d threads.\n", num_blocks.x);
    //Find best layout of all blocks
    champion_kernel<<<1, num_blocks>>>(best_mem, score_mem);
    short *host_pop = 0;
    short *host_best = 0;
    int *host_scores = 0;
    host_pop = (short*)malloc(total_mem);
    host_best = (short*)malloc((num_blocks.x+1) * per_copy_size);
    host_scores = (int*)malloc((num_blocks.x+1) * sizeof(int));

    //Copy data to host
    cudaMemcpy(host_pop, pop_mem, total_mem, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_best, best_mem, (num_blocks.x+1) * per_copy_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_scores, score_mem, (num_blocks.x+1) * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Best scores: [";
    for (unsigned int i=0; i<NUM_BLOCKS+1; ++i) {
	cout << host_scores[i] << ", ";
    }
    cout << "]" << endl;
    cout << "Best layout: [";
    int offset_best = NUM_BLOCKS * node_data[0];
    for (unsigned int i=0; i<node_data[0]; i+=2) {
	cout << "(" << host_best[offset_best+i] << ", " << host_best[offset_best+i+1] << "), ";
    }
    cout << "]" << endl;

    int best = host_scores[NUM_BLOCKS];
    if (best <= 0 || best >= 10000) {
	best = 10000;
    }
    cout << best;

    //Free memory
    free(host_pop);
    free(host_best);
    free(host_scores);
    
    cudaFree(pop_mem);
    cudaFree(temp_mem);
    cudaFree(best_mem);
    cudaFree(score_mem);
    cudaFree(rand_states);
    
}

