#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "windows.h"
#include <iostream>
#include <fstream>

using namespace std;

//Define matrix dimensions
#define N 0

//quick and dirty matrix from given values
//!! INITIALIZE MATRIX A TO GoL STARTING MATRIX HERE !!
int A[N][N];


int A_dst[N][N];

__global__ static void calcGrid(int *A, int *B)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int currPos = N * y + x;

    // Begin counting number of neighbors:
    int neighbors = 0;

    // While on top or bot of matrix
    if(currPos >= 0 && currPos <= (N-1)) //top row index
	{
		//B[currPos] = 1;
		if(currPos == 0)//if currPos is top left of matrix
		{
			int botRow = N*(N-1);
			int wrappedPos = botRow + currPos;

			if (A[wrappedPos+(N-1)] == 1) //arr[col-1][row-1] (top left) wraps to bot row, then wraps around to far right
				neighbors += 1;
			if (A[(currPos+(N-1))] == 1) //arr[col-1][row] (left)
				neighbors += 1;
			if (A[(currPos+N)+(N-1)] == 1) //arr[col-1][row+1] (bot left)
				neighbors += 1;
			if (A[wrappedPos] == 1) //arr[col][row-1] (top)
				neighbors += 1;
			if (A[currPos+N] == 1) //arr[col][row+1] (bot)
				neighbors += 1;
			if (A[wrappedPos+1] == 1) //arr[col+1][row-1] (top right)
				neighbors += 1;
			if (A[currPos+1] == 1) //arr[col+1][row] (right)
				neighbors += 1;
			if (A[(currPos+1)+N] == 1) //arr[col+1][row+1] (bot right)
				neighbors += 1;
		}
		else if(currPos == (N-1)) //if currPos is top right of matrix
		{
			int botRow = N*(N-1);
			int wrappedPos = botRow + currPos;

			if (A[wrappedPos-1] == 1) //arr[col-1][row-1] (top left)
				neighbors += 1;
			if (A[(currPos-1)] == 1) //arr[col-1][row] (left)
				neighbors += 1;
			if (A[(currPos-1)+N] == 1) //arr[col-1][row+1] (bot left)
				neighbors += 1;
			if (A[wrappedPos] == 1) //arr[col][row-1] (top)
				neighbors += 1;
			if (A[currPos+N] == 1) //arr[col][row+1] (bot)
				neighbors += 1;
			if (A[wrappedPos-(N-1)] == 1) //arr[col+1][row-1] (top right) wrap to bot, then wrap around to far left
				neighbors += 1;
			if (A[currPos-(N-1)] == 1) //arr[col+1][row] (right) 
				neighbors += 1;
			if (A[currPos+1] == 1) //arr[col+1][row+1] (bot right) wrap to far left col next row
				neighbors += 1;
		}
		else // 1 through (N-2) top row not first/last col
		{
			int botRow = N*(N-1);
			int wrappedPos = botRow + currPos;

			if (A[wrappedPos-1] == 1) //arr[col-1][row-1] (top left) wraps to bot row, then wraps around to far right
				neighbors += 1;
			if (A[(currPos-1)] == 1) //arr[col-1][row] (left)
				neighbors += 1;
			if (A[(currPos-1)+N] == 1) //arr[col-1][row+1] (bot left)
				neighbors += 1;
			if (A[wrappedPos] == 1) //arr[col][row-1] (top)
				neighbors += 1;
			if (A[currPos+N] == 1) //arr[col][row+1] (bot)
				neighbors += 1;
			if (A[wrappedPos+1] == 1) //arr[col+1][row-1] (top right)
				neighbors += 1;
			if (A[(currPos+1)] == 1) //arr[col-1][row] (right)
				neighbors += 1;
			if (A[(currPos+1)+N] == 1) //arr[col+1][row+1] (bot right)
				neighbors += 1;
		}
		
	}
	else if(currPos >= (N*(N-1)) && currPos < (N*N)) //on bot row
	{
		//B[currPos] = 1;
		if(currPos == (N*(N-1)))//if currPos is bot left of matrix
		{
			int topRow = 0;
			int wrappedPos = topRow + (currPos % N);
			 
			if (A[(currPos-N)+(N-1)] == 1) //arr[col-1][row-1] (top left) wraps around to far right of above row
				neighbors += 1;
			if (A[(currPos+(N-1))] == 1) //arr[col-1][row] (left)
				neighbors += 1;
			if (A[(wrappedPos)+(N-1)] == 1) //arr[col-1][row+1] (bot left)
				neighbors += 1;
			if (A[currPos-N] == 1) //arr[col][row-1] (top)
				neighbors += 1;
			if (A[wrappedPos] == 1) //arr[col][row+1] (bot)
				neighbors += 1;
			if (A[(currPos-N)+1] == 1) //arr[col+1][row-1] (top right)
				neighbors += 1;
			if (A[currPos+1] == 1) //arr[col+1][row] (right)
				neighbors += 1;
			if (A[(wrappedPos)+1] == 1) //arr[col+1][row+1] (bot right)
				neighbors += 1;
		}
		else if(currPos == ((N*N)-1)) //if currPos is bot right of matrix
		{
			int topRow = 0;
			int wrappedPos = topRow + (currPos % N);

			if (A[((currPos-N)-1)] == 1) //arr[col-1][row-1] (top left) wraps around to far right of above row
				neighbors += 1;
			if (A[(currPos-1)] == 1) //arr[col-1][row] (left)
				neighbors += 1;
			if (A[wrappedPos-1] == 1) //arr[col-1][row+1] (bot left)
				neighbors += 1;
			if (A[currPos-N] == 1) //arr[col][row-1] (top)
				neighbors += 1;
			if (A[wrappedPos] == 1) //arr[col][row+1] (bot)
				neighbors += 1;
			if (A[(currPos-N)-(N-1)] == 1) //arr[col+1][row-1] (top right)
				neighbors += 1;
			if (A[currPos-(N-1)] == 1) //arr[col+1][row] (right)
				neighbors += 1;
			if (A[(wrappedPos-(N-1))] == 1) //arr[col+1][row+1] (bot right)
				neighbors += 1;
		}
		else // ((N*(N-1))+1) through ((N*N)-2) bot row not first/last col
		{
			int topRow = 0;
			int wrappedPos = topRow + (currPos % N);

			if (A[(currPos-1)-N] == 1) //arr[col-1][row-1] (top left)
				neighbors += 1;
			if (A[(currPos-1)] == 1) //arr[col-1][row] (left)
				neighbors += 1;
			if (A[(wrappedPos-1)] == 1) //arr[col-1][row+1] (bot left) 
				neighbors += 1;
			if (A[currPos-N] == 1) //arr[col][row-1] (top)
				neighbors += 1;
			if (A[wrappedPos] == 1) //arr[col][row+1] (bot) 
				neighbors += 1;
			if (A[(currPos+1)-N] == 1) //arr[col+1][row-1] (top right)
				neighbors += 1;
			if (A[currPos+1] == 1) //arr[col+1][row] (right)
				neighbors += 1;
			if (A[(wrappedPos+1)] == 1) //arr[col+1][row+1] (bot right) 
				neighbors += 1;
		}
	}
	else //else not top or bot row
	{
		//B[currPos] = 0;

		if(currPos % N == 0)//if currPos is far left of matrix
		{
			if (A[(currPos-N)+(N-1)] == 1) //arr[col-1][row-1] (top left) wraps around to far right of above row
				neighbors += 1;
			if (A[(currPos+(N-1))] == 1) //arr[col-1][row] (left)
				neighbors += 1;
			if (A[(currPos+N)+(N-1)] == 1) //arr[col-1][row+1] (bot left)
				neighbors += 1;
			if (A[currPos-N] == 1) //arr[col][row-1] (top)
				neighbors += 1;
			if (A[currPos+N] == 1) //arr[col][row+1] (bot)
				neighbors += 1;
			if (A[(currPos-N)+1] == 1) //arr[col+1][row-1] (top right)
				neighbors += 1;
			if (A[currPos+1] == 1) //arr[col+1][row] (right)
				neighbors += 1;
			if (A[(currPos+N)+1] == 1) //arr[col+1][row+1] (bot right)
				neighbors += 1;
		}
		else if(currPos % N == (N-1)) //if currPos is far right of matrix
		{
			if (A[(currPos-N)-1] == 1) //arr[col-1][row-1] 
				neighbors += 1;
			if (A[currPos-1] == 1) //arr[col-1][row] (left)
				neighbors += 1;
			if (A[(currPos+N)-1] == 1) //arr[col-1][row+1] (bot left)
				neighbors += 1;
			if (A[currPos-N] == 1) //arr[col][row-1] (top)
				neighbors += 1;
			if (A[currPos+N] == 1) //arr[col][row+1] (bot)
				neighbors += 1;
			if (A[(currPos-N)-(N-1)] == 1) //arr[col+1][row-1] (top right)
				neighbors += 1;
			if (A[currPos-(N-1)] == 1) //arr[col+1][row] (right)
				neighbors += 1;
			if (A[(currPos+N)-(N-1)] == 1) //arr[col+1][row+1] (bot right)
				neighbors += 1;
		}
		else // ((N*(N-1))+1) through ((N*N)-2) bot row not first/last col
		{
			if (A[(currPos-N)-1] == 1) //arr[col-1][row-1] (top left) 
				neighbors += 1;
			if (A[currPos-1] == 1) //arr[col-1][row] (left)
				neighbors += 1;
			if (A[(currPos+N)-1] == 1) //arr[col-1][row+1] (bot left)
				neighbors += 1;
			if (A[currPos-N] == 1) //arr[col][row-1] (top)
				neighbors += 1;
			if (A[currPos+N] == 1) //arr[col][row+1] (bot)
				neighbors += 1;
			if (A[(currPos-N)+1] == 1) //arr[col+1][row-1] (top right)
				neighbors += 1;
			if (A[currPos+1] == 1) //arr[col+1][row] (right)
				neighbors += 1;
			if (A[(currPos+N)+1] == 1) //arr[col+1][row+1] (bot right)
				neighbors += 1;
		}
	}

    //Apply rules to the cell:
    if (A[currPos] == 1 && neighbors < 2)
		B[currPos] = 0;
    else if (A[currPos] == 1 && neighbors > 3)
		B[currPos] = 0;
    else if (A[currPos] == 1 && (neighbors == 2 || neighbors == 3))
		B[currPos] = 1;
    else if (A[currPos] == 0 && neighbors == 3)
		B[currPos] = 1;
	else 
		B[currPos] = 0;

}


int main()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	std::cout << "Using:" << deviceProp.name << std::endl;

	//init pointers/vars
	int *gpu_A, *gpu_B, *gpu_tmp , i,j;
	size_t pitch_a;
	
	//create output file
	ofstream myfile;
	myfile.open("./output.txt");
	myfile << "Using: " << deviceProp.name << "\n\n";

	//calc thread/blk size
	//allocate memory on gpu
	//copy input and dest array to gpu
	int size = N * N * sizeof(int);
	cudaMalloc(&gpu_A, size);
	cudaMalloc(&gpu_B, size);
	cudaMemcpy(gpu_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B, A_dst, size, cudaMemcpyHostToDevice);
		
	int threadsPerBlk = N;
	int blocksPerGrid = (N*N*sizeof(int) + threadsPerBlk - 1) /threadsPerBlk;

	for(int g=0;g<100;g++)
	{
		//allocate width*height (in bytes) of linear memory on the device. Returns in *devPtr a pointer to the allocated memory
		//cudaMallocPitch((void**)&gpu_A, &pitch_a, sizeof(int)*N, N);
		//copy matrix from host to device
		//cudaError_t myError = cudaMemcpy2D(gpu_A, pitch_a, A, sizeof(int)*N, sizeof(int)*N, N, cudaMemcpyHostToDevice);

		//GPU fn call
		calcGrid<<<blocksPerGrid, threadsPerBlk>>>(gpu_A, gpu_B);

		// Views each iteration, if left commented out will only print the final iteration 
		/*
		//copy matrix from device back to host
		cudaMemcpy(A, gpu_B, size, cudaMemcpyDeviceToHost);
		myfile << "\n\n---------------------------------------ITERATION: " << g << "---------------------------------------\n";
		for (int i=0; i<N; i++)
		{
			for(int j=0; j<N+1; j++)
			{
				if(j == N) 
				myfile << ("\n");
				else
					if(A[i][j]==1)
						myfile << "*";//alive
					else
						myfile << "-";//dead
			}
		}*/
		
		if(g != 99)
		{
			//swap pointers for next iteration
			gpu_tmp = gpu_A;
			gpu_A = gpu_B;
			gpu_B = gpu_tmp;
		}

	}

	//copy matrix from device back to host
	cudaMemcpy(A, gpu_B, size, cudaMemcpyDeviceToHost);
	
	for (int i=0; i<N; i++)
	{
		for(int j=0; j<N+1; j++)
		{
			if(j == N)
			myfile << ("\n");
			else
				if(A[i][j]==1)
					myfile << "*";//alive
				else
					myfile << "-";//dead
		}
	}

	myfile.close();

	cudaFree(gpu_A);
	cudaFree(gpu_B);
	return 0;
} 
