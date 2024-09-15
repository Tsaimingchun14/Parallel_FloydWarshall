#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>


//======================
#define DEV_NO 0
cudaDeviceProp prop;

const int INF = ((1 << 30) - 1);
int n, m, n_old;
int *Dist, *dDist[2];
#define B 32   //block size




void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n_old, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    n = (n_old + B - 1) / B * B;

    Dist = (int*)malloc(n*n*sizeof(int));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j && i < n_old) {
                Dist[i*n+j] = 0;
            } else {
                Dist[i*n+j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]*n + pair[1]] = pair[2];
    }
    fclose(file);

    
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n_old; ++i) {
        fwrite(&Dist[i*n], sizeof(int), n_old, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }


__global__ void phase1(int *dDist, int n, int r){
    __shared__ int s[B*B];

    int j = threadIdx.x;
    int i = threadIdx.y;

    // block id = (r,r)  p.s. block refer to partition in dist matrix, not kernel block 
    int actual_start_i = r * B;
    int actual_start_j = r * B;


    s[ i*B + j ] = dDist[ (i+actual_start_i)*n + (j+actual_start_j)];
    
    #pragma unroll
    for (int k = 0; k < B ; ++k){
        __syncthreads();
        s[ i*B + j ] = min( s[ i*B + j ] , s[ i*B + k ]+s[ k*B + j ]);
    }

    dDist[ (i+actual_start_i)*n + (j+actual_start_j)] = s[ i*B + j ];
}

__global__ void phase2(int *dDist, int n, int r, int round, bool row){

    __shared__ int s[B*B];
    __shared__ int block_r_r[B*B];

    int j = threadIdx.x;
    int i = threadIdx.y;

    int actual_start_i;
    int actual_start_j;

    
    if(row == 1){
        actual_start_i = r * B;
        actual_start_j = ((r+1 + blockIdx.x) % round) * B;
    }
    else{
        actual_start_j = r * B;
        actual_start_i = ((r+1 + blockIdx.x) % round) * B;
    }

    s[ i*B + j ] = dDist[ (i+actual_start_i)*n + (j+actual_start_j)];
    block_r_r[ i*B + j] = dDist[ (i+ r*B )*n + (j+ r*B )];  
    int tmp = dDist[ (i+actual_start_i)*n + (j+actual_start_j)];                 
    
    if(row == 1){
        #pragma unroll
        for (int k = 0; k < B ; ++k){
            
            __syncthreads();
            s[ i*B + j ] = tmp;
            tmp = min( s[ i*B + j ] , block_r_r[ i*B + k ]+s[ k*B + j ]);
        }
    }
    else{
        #pragma unroll
        for (int k = 0; k < B ; ++k){
            
            __syncthreads();
            s[ i*B + j ] = tmp;
            tmp = min( s[ i*B + j ] , s[ i*B + k ]+block_r_r[ k*B + j ]);
        }
    }
    

    dDist[ (i+actual_start_i)*n + (j+actual_start_j)] = tmp;
}

__global__ void phase3(int *dDist, int n, int r, int round, int device_id){

    int actual_start_i = ((r+1 + blockIdx.y) % round) * B;
    int actual_start_j = ((r+1 + blockIdx.x) % round) * B;

    if(device_id == 1){
        if(actual_start_i/B < round/2) return;
    }
    else{
        if(actual_start_i/B >= round/2) return;
    }

    __shared__ int block_x_r[B*B];
    __shared__ int block_r_x[B*B];

    int j = threadIdx.x;
    int i = threadIdx.y;
  
    

    block_x_r[ i*B + j ] = dDist[ (i+actual_start_i)*n + (j+r*B)];
    block_r_x[ i*B + j] = dDist[ (i+ r*B )*n + (j+ actual_start_j )];                       
    
    int tmp = dDist[ (i+actual_start_i)*n + (j+ actual_start_j)];

    #pragma unroll
    for (int k = 0; k < B ; ++k){
        __syncthreads();
        tmp = min( tmp , block_x_r[ i*B + k ]+block_r_x[ k*B + j ]);
    }
    
    dDist[ (i+actual_start_i)*n + (j+ actual_start_j)] = tmp;
    
}


int main(int argc, char* argv[]) {


    input(argv[1]);
    printf("n = %d\n",n_old);

    #pragma omp parallel num_threads(2)
    {
        int id = omp_get_thread_num();

        cudaSetDevice(id);

        cudaHostRegister(Dist, n*n*sizeof(int), cudaHostRegisterDefault);
    
        cudaMalloc(&dDist[id], n*n*sizeof(int));

        int round = n/B;

        int second_start = n*B*(round/2);
        int second_size = n*n - second_start;

        if(id == 0){                   //first size = second_start
            cudaMemcpy(dDist[0], Dist, second_start*sizeof(int), cudaMemcpyHostToDevice);
        }
        else{
            cudaMemcpy(dDist[1]+second_start, Dist+second_start, second_size*sizeof(int), cudaMemcpyHostToDevice);
        } 
        #pragma omp barrier

        for (int r = 0; r < round; ++r){

            if(id == 0){
                cudaMemcpyPeer(dDist[1], 1, dDist[0], 0, second_start*sizeof(int));
            }
            else{
                cudaMemcpyPeer(dDist[0]+second_start, 0, dDist[1]+second_start, 1, second_size*sizeof(int));
            }
            #pragma omp barrier
            phase1<<<1,dim3(B,B)>>>(dDist[id], n, r);
            cudaStream_t stream1, stream2;
            cudaStreamCreate(&stream1); cudaStreamCreate(&stream2);
            phase2<<< round -1 ,dim3(B,B),0,stream1>>>(dDist[id], n, r, round, 0);
            phase2<<< round -1 ,dim3(B,B),0,stream2>>>(dDist[id], n, r, round, 1);
            cudaDeviceSynchronize();
            phase3<<<dim3(round -1, round -1),dim3(B,B)>>>(dDist[id], n, r, round, id);
        }

        if(id == 0){
            cudaMemcpy(Dist, dDist[0],  second_start*sizeof(int), cudaMemcpyDeviceToHost);
        }
        else{
            cudaMemcpy(Dist+second_start, dDist[1]+second_start,  second_size*sizeof(int), cudaMemcpyDeviceToHost);
        }
        #pragma omp barrier

    }
   
    output(argv[2]);
    return 0;
}

