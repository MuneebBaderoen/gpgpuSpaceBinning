#ifndef GPUPATHFIND_CUH
#define GPUPATHFIND_CUH

#include <cuda_runtime.h>
#include <cuda.h>
#include <string>

__global__ void cudaAllocateAsteroidToBin(){
	//int id = blockIdx.x*blockDimx+blockIDx.y;

}

__global__ void cudaFindLocalPath(){

}

void checkError(cudaError_t errorBool, std::string message){
	using namespace std;
	if(!errorBool)
		cout<<"Error: ";
	else
		cout<<"Passed: ";
		
	cout<<message<<endl;
	cint.get();
}





#endif