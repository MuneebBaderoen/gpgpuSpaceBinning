#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>


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

__constant__ long dc_numAsteroids;
__constant__ float dc_stepSize;
__constant__ int dc_gridx;
int gridXtoDevice= 0;





//Global declarations
long numAsteroids;
float stepSize = 10;

float * h_path;
//__device__ float * d_path;

//Host variable declarations
Asteroid * h_asteroids;
float * h_bins; //Bin x is value, y is direction for path calculation
float * h_compareToGPU;

Point h_ship;
Point h_baseStation;
Point h_gridSize;

Point d_ship;
Point d_baseStation;
Point d_gridSize;

//cuda stuff
cudaError_t result;


//--------------------INPUT
void readFile(const char* filename){
	using namespace std;

	ifstream inFile(filename, ios::in | ios::binary);
	//string s = "";
	if(inFile.is_open()){
		inFile.seekg (0, ios::beg);
		//read base station position
		float f;
		inFile.read(reinterpret_cast<char*>(&h_baseStation.x), sizeof(float));
		inFile.read(reinterpret_cast<char*>(&h_baseStation.y), sizeof(float));
		inFile.read(reinterpret_cast<char*>(&f), sizeof(float));

		//get number of asteroids
		int pos = inFile.tellg ();		

		inFile.seekg (0, ios::end);
   		numAsteroids = (long)(((int)inFile.tellg())-pos)/12;   		
    	inFile.seekg (pos, ios::beg);

		//initialize array
		h_asteroids = new Asteroid [numAsteroids];

		int count = 0;

		//populate array
		while(inFile.good()){	
			float xVal;
			inFile.read(reinterpret_cast<char*>(&xVal), sizeof(float));
			float yVal;
			inFile.read(reinterpret_cast<char*>(&yVal), sizeof(float));
			float val;
			inFile.read(reinterpret_cast<char*>(&val), sizeof(float));
			h_asteroids[count++]=Asteroid(xVal, yVal, val);
			//cout<<count-1<<": "<<asteroids[count-1]<<endl;
			//cout<<"Float val: "<<f<<" "<<2*f<<endl;
			

		}
	}

	inFile.close();
}

//--------------------OUTPUT TO CONSOLE
void binPrint(float * a){
	using namespace std;
	for(int i = 0; i<h_gridSize.x; ++i){
		for(int j = 0; j<h_gridSize.y; ++j){
			cout<<a[(int)(i*h_gridSize.x+j)]<<", ";
		}
		cout<<endl;
	}
}

void pathPrint(float * a){
	using namespace std;

	vector<Point> onPath;

	for(int i = 0; i<2*2*h_gridSize.x-1; i+=2){
		float x = h_path[i], y=h_path[i+1];
		onPath.push_back(Point(x,y));
	}

	for(int i = 0; i<h_gridSize.x; ++i){
		for(int j = 0; j<h_gridSize.y; ++j){
			vector<Point>::iterator it = find(onPath.begin(), onPath.end(), Point(i,j));
			if(it!=onPath.end())
				cout<<"1, ";
			else
				cout<<"0, ";
		}
		cout<<endl;
	}
}

//--------------------BIN INITIALIZATION
void binInitialization(){
	using namespace std;

	//calculate grid size
	h_gridSize.x=(int)(h_baseStation.x+stepSize/2)/(int) stepSize + 1;
	h_gridSize.y=(int)(h_baseStation.y+stepSize/2)/(int) stepSize + 1;

	//allocate memory
	h_compareToGPU = new float[int(h_gridSize.x*h_gridSize.y)];
	h_bins = new float[int(h_gridSize.x*h_gridSize.y)];
	h_path = new float[int(2*(2*h_gridSize.x-1))];

	//initialize to 0
	for(int i = 0; i< h_gridSize.x*h_gridSize.y;++i){
		h_bins[i]=0;	
		h_compareToGPU[i]=0;
	}

	for(int i = 0; i< 2*(2*h_gridSize.x-1);++i)
		h_path[i]=0;		
}

void binInitializationTest(){
	using namespace std;

	binInitialization();

	//initialize to 0
	for(int i = 0; i< h_gridSize.x*h_gridSize.y;++i)
		h_bins[i]=1;	

	h_bins[(int)(5*h_gridSize.x+5)]=5;	
}

//--------------------CPU FUNCTIONS

void cpuSequentialBinning(){
	using namespace std;
	Point binId, binPos;
	
	for(int i = 0; i<numAsteroids; ++i){	

		binId.x=(int)(h_asteroids[i].x+stepSize/2)/(int) stepSize;
		binId.y=(int)(h_asteroids[i].y+stepSize/2)/(int) stepSize;

		binPos.x=binId.x*stepSize;
		binPos.y=binId.y*stepSize;

		float deltaX = h_asteroids[i].x-binPos.x;
		float deltaY = h_asteroids[i].y-binPos.y;
		
		if((deltaX*deltaX+deltaY*deltaY)<stepSize*stepSize/4){			
			h_bins[(int)(binId.y*h_gridSize.x+binId.x)]+=h_asteroids[i].value;
		}		
	}	
}

void cpuValuePropagation(){
	using namespace std;
	for(int i = 0; i<h_gridSize.x; ++i){
		for(int j = 0; j<h_gridSize.y; ++j){	
			 
			double sumup   = (j-1>=0)?h_bins[(int)((j-1)*h_gridSize.x+i)]:0;				
			double sumleft = (i-1>=0)?h_bins[(int)(j*h_gridSize.x+(i-1))]:0;						

			//h_bins[(int)(j*h_gridSize.x+i)].y = (((sumleft-sumup)>0)?1:-1);
			h_bins[(int)(j*h_gridSize.x+i)]+= max(sumleft, sumup);

			//cout<<"Direction set: "<<h_bins[(int)(j*h_gridSize.x+i)].y<<endl;
			//cout<<(int)(j*h_gridSize.x+i)<<" max is from: "<<(h_bins[(int)(j*h_gridSize.x+i)].y==-1?"above":"left")<<endl;
		

			
		}
	}	
}

void cpuGetPath(){
	

	using namespace std;

	Point currentPoint;
	Point nextIndex(h_gridSize.x-1, h_gridSize.x-1);
	int count = 0;
	do{		
		double sumup   = (nextIndex.y-1>=0)?h_bins[(int)((nextIndex.y-1)*h_gridSize.x+nextIndex.x)]:0;				
		double sumleft = (nextIndex.x-1>=0)?h_bins[(int)(nextIndex.y*h_gridSize.x+(nextIndex.x-1))]:0;

		int nextDir = (max(sumleft,sumup)==sumleft)?1:-1;
		//h_bins[(int)(j*h_gridSize.x+i)].y = (((sumleft-sumup)>0)?1:-1);
		currentPoint.x=nextIndex.x;
		currentPoint.y=nextIndex.y;

		if(nextDir==1){nextIndex.x-=1;}
		if(nextDir==-1){nextIndex.y-=1;}
		//cout<<"current  : "<<currentPoint<<endl;
		
		h_path[count++]=currentPoint.x;
		h_path[count++]=currentPoint.y;

		//cout<<"direction: "<<((nextDir==1)?"left":"up")<<endl; 
		//cout<<"next     : "<<endl<<endl;
		
	}while(currentPoint.x!=0 || currentPoint.y!=0);
}

//--------------------GPU FUNCTIONS

__global__ void gpuAllocateAsteroidToBin(Asteroid* asteroids, float* bins){

	
	bins[0]=dc_stepSize;
	bins[1]=dc_gridx;
	bins[2]=dc_numAsteroids;
	/*
	int i_x = blockIdx.x * blockDim.x + threadIdx.x;
	int i_y = blockIdx.y * blockDim.y + threadIdx.y;
	

	int pitch = gridDim.x * blockDim.x;
	

	int i = i_y * pitch + i_x;
	
	if(i<dc_numAsteroids){
	
		float binIdx=(int)(asteroids[i].x+dc_stepSize/2)/(int) dc_stepSize;
		float binIdy=(int)(asteroids[i].y+dc_stepSize/2)/(int) dc_stepSize;

		float binPosx=binIdx*dc_stepSize;
		float binPosy=binIdy*dc_stepSize;

		float deltaX = asteroids[i].x-binPosx;
		float deltaY = asteroids[i].y-binPosy;
		
		if((deltaX*deltaX+deltaY*deltaY)<dc_stepSize*dc_stepSize/4){			
			bins[(int)(binIdy*dc_gridx+binIdx)]+=asteroids[i].value;
		}	
	}
	*/
	
	
}

__global__ void gpuPropagateMaxValues(){

	
	//int i = threadIdx.x + blockIdx.x * blockDim.x;
	//int j = threadIdx.y + blockIdx.y * blockDim.y;

	//int index = gridDim.x*j+i;

	int i = threadIdx.x;//*blockIdx.x+ ;
	int j = threadIdx.y;

	//if(index<dc_gridx){



	//}
	
	
	//int idUp = (j-1)*dc_gridx+i;
	//int idLeft = j*dc_gridx+i-1;

	//double sumup   = (j-1>=0)?d_bins[idUp]:0;				
	//double sumleft = (i-1>=0)?d_bins[idLeft]:0;						

	
	//d_bins[id]+= ((sumleft - sumup)>0)?sumleft:sumup;

	//d_bins[id]=id;

	/*
	int id = j*dc_gridx+i;
	int idUp = (j-1)*dc_gridx+i;
	int idLeft = j*dc_gridx+i-1;
	
	if(i+j==level){
		double sumup   = (j-1>=0)?d_bins[idUp]:0;				
		double sumleft = (i-1>=0)?d_bins[idLeft]:0;						

		//d_bins[id].y = (((sumleft-sumup)>0)?1:-1);
		d_bins[id]+= ((sumleft - sumup)>0)?sumleft:sumup;


	}
	*/


	


}

__global__ void gpuFindLocalPath(){

}

void checkError(cudaError_t errorBool, std::string message){
	using namespace std;
	
	if(errorBool!=cudaSuccess){
		cout<<errorBool<<endl;
		cout<<"Error: ";
	}
	else
		cout<<"Passed: ";
		
	cout<<message<<"  -   "<<cudaGetErrorString(errorBool)<<endl;
	//cin.get();
}

void gpuInitialization(){
	//result = cudaSetDevice(0);
	result = cudaFree(NULL);
	
		
	//result=cudaMalloc(&dc_stepSize, sizeof(float));
	//result=cudaMalloc(dc_numAsteroids, sizeof(long));
	//result=cudaMalloc(&dc_gridx, sizeof(float));
	gridXtoDevice = (int)(h_gridSize.x);
	result = cudaMemcpyToSymbol(dc_gridx, &gridXtoDevice, sizeof(int),0,cudaMemcpyHostToDevice);
	checkError(result, "Copying gridSize to symbol");

	result = cudaMemcpyToSymbol(dc_stepSize, &stepSize, sizeof(float), 0, cudaMemcpyHostToDevice);
	checkError(result, "Copying stepsize to symbol");	

	result = cudaMemcpyToSymbol(dc_numAsteroids, &numAsteroids, sizeof(long),0, cudaMemcpyHostToDevice);
	checkError(result, "Copying num asteroid to symbol");
	

	

	
	

	//Allocations for binning
	result=cudaMalloc(&d_asteroids, numAsteroids*sizeof(Asteroid));
	checkError(result, "Allocating Asteroid memory");

	result = cudaMemcpy(d_asteroids, h_asteroids, numAsteroids*sizeof(Asteroid), cudaMemcpyHostToDevice);
	checkError(result, "Copying asteroid Data");


	result=cudaMalloc(&d_bins, h_gridSize.x*h_gridSize.x*sizeof(float));
	checkError(result, "Allocating Bin memory");

	result = cudaMemcpy(d_bins, h_bins, h_gridSize.x*h_gridSize.x*sizeof(float), cudaMemcpyHostToDevice);
	checkError(result, "Copying bin data to device");
	
}

void gpuParallelBinning(){
	
	int numThreads = numAsteroids;
	int threadsPerBlock = 1024;
	int numBlocks = ceil((float)numAsteroids/1024);
	int cudaGridx = ceil(sqrtf(numBlocks));

	std::cout<<"numblocks: "<<numBlocks<<", gridX: "<<cudaGridx<<std::endl;

	dim3 cudaBlockSize(32,32,1);
	//dim3 cudaGridSize(1024, 1024);
	dim3 cudaGridSize(cudaGridx, cudaGridx,1);

	
	//gpuAllocateAsteroidToBin<<<cudaGridSize,cudaBlockSize>>>(d_asteroids, d_bins);
	gpuAllocateAsteroidToBin<<<1,1>>>(d_asteroids, d_bins);
	cudaThreadSynchronize();
	//checkError();

}

void gpuValuePropagation(){
	/*
	int numThreads = h_gridSize.x;
	int threadsPerBlock = 1024;
	int numBlocks = ceil((float)numThreads/threadsPerBlock);
	int cudaGridx = ceil(sqrtf(numBlocks));

	std::cout<<"numblocks: "<<numBlocks<<", gridX: "<<cudaGridx<<std::endl;

	dim3 cudaBlockSize(32,32,1);	
	dim3 cudaGridSize(cudaGridx, cudaGridx,1);
	*/

	dim3 cudaBlockSize(11,1,1);
	//dim3 cudaGridSize(1024, 1024);
	dim3 cudaGridSize(1,1,1);
	
	//for(int i = 0; i< 2*h_gridSize.x-1;i++){
	int i = 11;
	gpuPropagateMaxValues<<<cudaGridSize,cudaBlockSize>>>();
	cudaThreadSynchronize();
	//}
}

void gpuCopyDataBack(){
	using namespace std;
	result = cudaMemcpy(h_compareToGPU, d_bins, (int)(h_gridSize.x*h_gridSize.x*sizeof(float)), cudaMemcpyDeviceToHost);
	checkError(result,"Copying bins data back from gpu");
	//cout<<h_compareToGPU[0]<<":"<<h_compareToGPU[1]<<":"<<h_compareToGPU[2]<<endl;

	binPrint(h_compareToGPU);
	unsigned long long testNumAst = 0;

	result = cudaMemcpyFromSymbol(&testNumAst, dc_numAsteroids, sizeof(long), 0,cudaMemcpyDeviceToHost);
	checkError(result,"Copying numast back from gpu");

	cout<<"Numast:"<<testNumAst<<endl;

	cudaFree(d_asteroids);
	cudaFree(d_bins);//Bin x is value, y is direction for path calculation


}



int main(int argc, char** argv){
	printf("num %i\n", 8);
	using namespace std;
	string fname = "s_data_02.scan";
	

	readFile(fname.c_str());
	cout<<"ShipPosition: "<<h_ship<<endl;
	cout<<"BasePosition: "<<h_baseStation<<endl;


	//binInitialization();
	binInitializationTest();

	//CPU implementation
	
	//cpuSequentialBinning();	
	//cpuValuePropagation();
	binPrint(h_bins);
	//cpuGetPath();
	//cout<<"Printing path now: "<<endl;
	//pathPrint(h_path);



	//GPU implementation
	gpuInitialization();
	gpuParallelBinning();
	//gpuValuePropagation();
	gpuCopyDataBack();
	

	cout<<"run complete"<<endl;

	cin.get();
	return 0;

}