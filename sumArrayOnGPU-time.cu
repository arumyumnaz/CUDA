#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

void initialData(float* ip, int size){
  time_t t;
  srand((unsigned int)time(&t));
  for(int i = 0; i<size; i++){
    ip[i] = (float)(rand() & 0xFF)/10.0f;
  }
}

void checkResult(float *hostRef, float *gpuRef, const int N){
  double epsilon = 1.0E-8;
  bool match = 1;
  for(int i = 0; i<N; i++){
    if(abs(hostRef[i] - gpuRef[i]) > epsilon){
      match = 0;
      printf("Arrays do not match!\n");
      printf("Host %5.2f GPU %5.2f at current %d", hostRef[i], gpuRef[i], i);
      break;
    }
  }
  if(match) printf("Arrays match. \n \n");
}

double cpuSecond(){
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void sumArraysOnHost(float* A, float* B, float* C, const int N){
  for(int idx = 0; idx<N; idx++){
    C[idx] = A[idx] + B[idx];
  }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N) C[i] = A[i] + B[i];
}


int main(int argc, char **argv){
  printf("%s Starting... \n", argv[0]);

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("Using Device %d:%s\n", dev, deviceProp.name);
  cudaSetDevice(dev);

  // set up date size of vectors
  int nElem = 1<<24;
  printf("Vector size %d\n", nElem);

  // malloc host memory
  size_t nBytes = nElem * sizeof(float);

  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A     = (float *)malloc(nBytes);
  h_B     = (float *)malloc(nBytes);
  hostRef = (float *)malloc(nBytes);
  gpuRef  = (float *)malloc(nBytes);

  double iStart, iElaps;

  // initialize data at host side
  iStart = cpuSecond();
  initialData(h_A, nElem);
  initialData(h_B, nElem);

  memset(hostRef, 0, nBytes);
  memset(gpuRef, 0, nBytes);

  // add vector at host side for result checks

  iStart = cpuSecond();
  sumArraysOnHost(h_A, h_B, hostRef, nElem);
  iElaps = cpuSecond() - iStart;

  // malloc device global memory
  float *d_A, *d_B, *d_C;
  cudaMalloc((float **)&d_A, nBytes);
  cudaMalloc((float **)&d_B, nBytes);
  cudaMalloc((float **)&d_C, nBytes);

  // transfer data from host to device
  cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

  // invoke kernel at host side
  int iLen = 1024;
  dim3 block(iLen);
  dim3 grid((nElem+block.x-1)/block.x);

  iStart = cpuSecond();
  sumArraysOnGPU<<< grid, block >>>(d_A, d_B, d_C, nElem);
  cudaDeviceSynchronize();
  iElaps = cpuSecond() - iStart;
  printf("sumArraysOnGPU <<< %d, %d >>> Time elapsed %f sec\n", grid.x, block.x, iElaps);

  // copy kernel result back to host side
  cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

  // check device results
  checkResult(hostRef, gpuRef, nElem);

  // free device global memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // free host memory
  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  return 0;
}
