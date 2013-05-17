#include <iostream>
#include <fstream>
#include <string>
//#include <cuda.h>
//#include "gpuPathfind.cuh"

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
};

std::ostream& operator<<(std::ostream&out, const Point& pt){
	out<<"Point: ("<<pt.x<<", "<<pt.y<<")";
	return out;
}




//Global declarations
unsigned long long numAsteroids;
float stepSize = 10;

float * h_path;

//Host variable declarations
Asteroid * h_asteroids;
float * h_bins; //Bin x is value, y is direction for path calculation

Point h_ship;
Point h_baseStation;
Point h_gridSize;


//Device variable declarations
Asteroid * d_asteroids;
float * d_bins;//Bin x is value, y is direction for path calculation

Point d_ship;
Point d_baseStation;
Point d_gridSize;

//cuda stuff
//cudaError_t result;

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
   		numAsteroids = (unsigned long long)(((int)inFile.tellg())-pos)/12;   		
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

void binPrint(float * a){
	using namespace std;
	for(int i = 0; i<h_gridSize.x; ++i){
		for(int j = 0; j<h_gridSize.y; ++j){
			cout<<a[(int)(j*h_gridSize.x+i)]<<", ";
		}
		cout<<endl;
	}
}

void binInitialization(){
	using namespace std;

	//calculate grid size
	h_gridSize.x=(int)(h_baseStation.x+stepSize/2)/(int) stepSize + 1;
	h_gridSize.y=(int)(h_baseStation.y+stepSize/2)/(int) stepSize + 1;

	//allocate memory
	h_bins = new float[int(h_gridSize.x*h_gridSize.y)];
	h_path = new float[int(h_gridSize.x*h_gridSize.y)];

	//initialize to 0
	for(int i = 0; i< h_gridSize.x*h_gridSize.y;++i)
		h_bins[i]=0;	

	for(int i = 0; i< h_gridSize.x*h_gridSize.y;++i)
		h_path[i]=0;		
}

void binInitializationTest(){
	using namespace std;

	//calculate grid size
	h_gridSize.x=(int)(h_baseStation.x+stepSize/2)/(int) stepSize + 1;
	h_gridSize.y=(int)(h_baseStation.y+stepSize/2)/(int) stepSize + 1;

	//allocate memory
	h_bins = new float[int(h_gridSize.x*h_gridSize.y)];
	h_path = new float[int(h_gridSize.x*h_gridSize.y)];

	//initialize to 0
	for(int i = 0; i< h_gridSize.x*h_gridSize.y;++i)
		h_bins[i]=1;	

	for(int i = 0; i< h_gridSize.x*h_gridSize.y;++i)
		h_path[i]=0;	

	h_bins[(int)(5*h_gridSize.x+5)]=5;	
}



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
	//binPrint(h_bins);

	using namespace std;

	Point currentPoint;
	Point nextIndex(h_gridSize.x-1, h_gridSize.x-1);
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
		h_path[(int)(currentPoint.y*h_gridSize.x+currentPoint.x)]=1;

		cout<<"direction: "<<((nextDir==1)?"left":"up")<<endl; 
		//cout<<"next     : "<<endl<<endl;
		
	}while(currentPoint.x!=0 || currentPoint.y!=0);

	cout<<"Printing path now: "<<endl;
	binPrint(h_path);


}

/*
void gpuInitialization(){
	cudaSetDevice(0);
	cudaFree(NULL);
}

void gpuParallelBinning(){

	checkError()

}*/


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
	
	cpuSequentialBinning();	
	cpuValuePropagation();
	cpuGetPath();

	//GPU implementation

	//gpuValuePropagationTest();
	//gpuParallelBinning();

	cout<<"run complete"<<endl;

	cin.get();
	return 0;

}