#include <cuda_runtime.h>
#include <stdio.h>
#include "initialData.h"
#include "checkResult.h"
#include "cpuSecond.h"

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny){
  float *ia = A;
  float *ib = B;
  float *ic = C;

  for(int iy=0; iy<ny; iy++){
    for(int ix=0; ix<nx; ix++){
      ic[ix] = ia[ix]+ib[ix];
    }
    ia += nx; ib += nx; ic += nx;
  }
}

__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC, int nx, int ny){
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = blockIdx.y;
  unsigned int idx = iy*nx + ix;
  if(ix < nx){
    MatC[idx] = MatA[idx] + MatB[idx];
  }
}
int main(int argc, char **argv){
  printf("%s Starting...\n", argv[0]);

  //get device information
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("Using Device %d:%s\n", dev, deviceProp.name);
  cudaSetDevice(dev);

  //set up data size of matrix
  int nx = 1<<14;
  int ny = 1<<14;

  int nxy = nx*ny;
  int nBytes = nxy*sizeof(float);
  printf("Matrix size: nx %d ny %d \n", nx, ny);

  //malloc host memory
  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  hostRef = (float *)malloc(nBytes);
  gpuRef = (float *)malloc(nBytes);

  //initialize data at host side
  double iStart = cpuSecond();
  initialData(h_A, nxy);
  initialData(h_B, nxy);
  double iElaps = cpuSecond() - iStart;

  memset(hostRef, 0, nBytes);
  memset(gpuRef, 0, nBytes);

  //add matrix at host side for result checks
  iStart = cpuSecond();
  sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
  iElaps = cpuSecond() - iStart;

  //malloc device global memory
  float *d_MatA, *d_MatB, *d_MatC;
  cudaMalloc((void **)&d_MatA, nBytes);
  cudaMalloc((void **)&d_MatB, nBytes);
  cudaMalloc((void **)&d_MatC, nBytes);

  //transfer data from host to device
  cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

  //invoke kernel at host side
  int dimx = 256;
  dim3 block(dimx, 1);
  dim3 grid((nx+block.x-1)/block.x , ny);

  iStart = cpuSecond();
  sumMatrixOnGPUMix<<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);
  cudaDeviceSynchronize();
  iElaps = cpuSecond() - iStart;
  printf("sumMatrixOnGPUMix<<< (%d, %d), (%d, %d) >>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

  //copy kernel result back to host side
  cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

  //check device results
  checkResult(hostRef, gpuRef, nxy);

  //free device global memory
  cudaFree(d_MatA);
  cudaFree(d_MatB);
  cudaFree(d_MatC);

  //free host memory
  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  //reset device
  cudaDeviceReset();
  return 0;
}
