#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<float.h>

#define index(i, j, N)  ((i)*(N)) + (j)
#define blockSize 256

void checkCudaSuccess(cudaError_t err, char *str){
    if(err != cudaSuccess)
    {
        fprintf(stderr," %s (error code %s)\n",str,cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  }
  
 __device__ void warpReduce(volatile float *sdata, unsigned int tid) 
  {
      sdata[tid] = min(sdata[tid],sdata[tid+32]);
      sdata[tid] = min(sdata[tid],sdata[tid+16]);
      sdata[tid] = min(sdata[tid],sdata[tid+8]);
      sdata[tid] = min(sdata[tid],sdata[tid+4]);
      sdata[tid] = min(sdata[tid],sdata[tid+2]);
      sdata[tid] = min(sdata[tid],sdata[tid+1]);
      
  }  

__global__ void findMinOfArray(float *matrix,int N, float *op_data,int num_blocks){
    
    unsigned int unique_id = blockIdx.x * blockDim.x + threadIdx.x; /* unique id for each thread in the block*/
    unsigned int row = unique_id % N; /*row number in the matrix*/
    unsigned int col = unique_id / N; /*col number in the matrix*/

    unsigned int thread_id = threadIdx.x; /* thread index in the block*/

    __shared__ float minChunk[blockSize];

    if((row >= 0) && (row < N) && (col >= 0) && (col < N)){
        minChunk[thread_id] = matrix[index(row,col,N)];
    }

    __syncthreads();

    for(unsigned int stride = (blockDim.x/2); stride > 32 ; stride /=2){
        __syncthreads();

        if(thread_id < stride)
        {
            minChunk[thread_id] = min(minChunk[thread_id],minChunk[thread_id + stride]);
        }
    }

    if(thread_id < 32){
        warpReduce(minChunk,thread_id);
    }

    if(thread_id == 0){
        op_data[index(0,blockIdx.x,num_blocks)] = minChunk[0];
    }
}

int main(int argc, char **argv){

    int N = atoi(argv[1]);

    double c_time_taken,d_time_taken;
    clock_t start, end;

    size_t matrix_size = N*N*sizeof(float);

    float *matrix;
    
    matrix = (float *) malloc(matrix_size);

    cudaError_t err = cudaSuccess;

    if(!matrix)
    {
        fprintf(stdout," memory not allocated");
        exit(1);
    }

    int i,j;

    fprintf(stdout,"start with filling matrix with random values\n");
    
    for(i = 0; i< N; i++){
        for(j=0;j<N;j++){
            matrix[index(i,j,N)] = (rand()%1000)*1.0 + 6;
        }
    }

    fprintf(stdout,"the matrix is filled with random values\n");

    float *d_matrix;     

    err = cudaMalloc((void **)&d_matrix,matrix_size);

    checkCudaSuccess(err,"Failed to allocate device memory for matrix");

    start = clock();

    err = cudaMemcpy(d_matrix,matrix,matrix_size,cudaMemcpyHostToDevice);

    checkCudaSuccess(err,"Failed to copy matrix from host to device");

    unsigned int block_size = blockSize;
    unsigned int grid_size = ceil((N*N)/(block_size*1.0));

    float *op_data = (float *)malloc(grid_size*sizeof(float));
    float *d_op_data;

    err = cudaMalloc((void **)&d_op_data,grid_size*sizeof(float));

    checkCudaSuccess(err,"Failed to allocate output memory for matrix");

    fprintf(stdout,"device compute started\n");

    findMinOfArray<<<grid_size,block_size>>>(d_matrix,N,d_op_data,grid_size);

    fprintf(stdout,"device compute done\n");

    err = cudaMemcpy(op_data,d_op_data,grid_size*sizeof(float),cudaMemcpyDeviceToHost);

    checkCudaSuccess(err,"Failed to copy output matrix from device to host ");
    
    float min_i = FLT_MAX;

    for(i = 0; i< grid_size;i++)
    {
        /*fprintf(stdout,"iter %d value is %d\n",i,op_data[index(0,i,grid_size)]);*/
        min_i = min(op_data[index(0,i,grid_size)],min_i);
    }

    end = clock();

    fprintf(stdout,"the min value in the array is %d\n", min_i);

    d_time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;

    printf("Time taken to compute min of the array with gpu is %lf\n", d_time_taken);

    start = clock();

    float c_min = FLT_MAX;
    for(i = 0; i< N; i++){
        for(j=0;j<N;j++){
            c_min = min(matrix[index(i,j,N)],c_min);
        }
    }

    end = clock();

    c_time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;

    printf("Time taken to compute min of the array with CPU is %lf\n", c_time_taken);

    printf("The speed up achieved is %f\n",c_time_taken/d_time_taken);
    
}