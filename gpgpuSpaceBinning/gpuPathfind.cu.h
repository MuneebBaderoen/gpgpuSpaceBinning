#ifndef GPUPATHFIND_CUH
#define GPUPATHFIND_CUH

#include <cuda_runtime.h>
#include <cuda.h>
#include <string>



struct Asteroid{
	Asteroid(){}
	Asteroid(float x1, float y1, float z1): x(x1) , y(y1) , value(z1){}
	float x, y, value;

};

std::ostream& operator<<(std::ostream&out, const Asteroid& ast){
	out<<"Asteroid: ("<<ast.x<<", "<<ast.y<<", "<<ast.value<<")";
	return out;
}

struct Point{
	Point(){}
	Point(float x1, float y1):x(x1),y(y1){}
	Point(int x1, int y1):x(x1),y(y1){}
	float x, y;

	bool operator==(const Point & p2){
		
		return (x==p2.x&&y==p2.y)?true:false;
	}
};

std::ostream& operator<<(std::ostream&out, const Point& pt){
	out<<"Point: ("<<pt.x<<", "<<pt.y<<")";
	return out;
}

//Device variable declarations
__device__ Asteroid * d_asteroids;
__device__ float * d_bins;//Bin x is value, y is direction for path calculation

__device__ __constant__ unsigned long long * dc_numAsteroids;
__device__ __constant__ float * dc_stepSize;


__global__ void cudaAllocateAsteroidToBin(Asteroid* asteroids, float * bins){

	bins[0]=asteroids[0].value;

	/*int i = blockIdx.x*blockDimx+blockIDx.y;

	
	
	
	float binIdx=(int)(asteroids[i].x+stepSize/2)/(int) stepSize;
	float binIdy=(int)(asteroids[i].y+stepSize/2)/(int) stepSize;

	float binPosx=binIdx*stepSize;
	float binPosy=binIdy*stepSize;

	float deltaX = asteroids[i].x-binPosx;
	float deltaY = asteroids[i].y-binPosy;
		
	if((deltaX*deltaX+deltaY*deltaY)<stepSize*stepSize/4){			
		h_bins[(int)(binIdy*h_gridSize.x+binIdx)]+=h_asteroids[i].value;
	}	
	*/
	
}

__global__ void cudaFindLocalPath(){

}

void checkError(cudaError_t errorBool, std::string message){
	using namespace std;
	
	if(errorBool!=cudaSuccess){
		cout<<errorBool<<endl;
		cout<<"Error: ";
	}
	else
		cout<<"Passed: ";
		
	cout<<message<<endl;
	//cin.get();
}





#endif