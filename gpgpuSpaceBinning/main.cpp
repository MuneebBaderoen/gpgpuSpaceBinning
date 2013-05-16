#include <iostream>
#include <fstream>
#include <string>
#include <cuda.h>
#include "gpuPathfind.cuh"

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

//Host variable declarations
Asteroid * h_asteroids;
float * h_bins;
Point h_ship;
Point h_baseStation;
Point h_gridSize;


//Device variable declarations
Asteroid * d_asteroids;
float * d_bins;
Point d_ship;
Point d_baseStation;
Point d_gridSize;


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
		h_asteroids = new Asteroid [numAsteroids*sizeof(Asteroid)];

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

void binInitialization(){
	using namespace std;

	//calculate grid size
	h_gridSize.x=(int)(h_baseStation.x+stepSize/2)/(int) stepSize + 1;
	h_gridSize.y=(int)(h_baseStation.y+stepSize/2)/(int) stepSize + 1;

	//allocate memory
	h_bins = new float[int(h_gridSize.x*h_gridSize.y)];

	//initialize to 0
	for(int i = 0; i< h_gridSize.x*h_gridSize.y;++i)
		h_bins[i]=0;

	
	
}

void cpuSequentialBinning(){
	using namespace std;
	Point binId, binPos;

	
	for(int i = 0; i<numAsteroids; ++i){
		//Point p (h_asteroids[i].x, h_asteroids[i].y);

		//cout<<h_asteroids[i]<<endl;

		binId.x=(int)(h_asteroids[i].x+stepSize/2)/(int) stepSize;
		binId.y=(int)(h_asteroids[i].y+stepSize/2)/(int) stepSize;

		//cout<<"BinID: "<<binId<<"bins index: "<<(binId.y*h_gridSize.x+binId.x)<<endl;
		
		//if((int)(binId.y*h_gridSize.x+binId.x)>h_gridSize.x*h_gridSize.x-1)
		//	continue;

		binPos.x=binId.x*stepSize;
		binPos.y=binId.y*stepSize;

		float deltaX = h_asteroids[i].x-binPos.x;
		float deltaY = h_asteroids[i].y-binPos.y;
		
		if((deltaX*deltaX+deltaY*deltaY)<stepSize){
			//cout<<(int)(binId.y*h_gridSize.x+binId.x)<<endl;
			h_bins[(int)(binId.y*h_gridSize.x+binId.x)]+=h_asteroids[i].value;
		}

		//cout<<h_bins[(int)(binPos.y*h_gridSize.x+binPos.x)]<<":"<<h_asteroids[i].value<<endl;

		//cout<<binPos<<endl;

	}
	//cout<<"counter: "<<counter<<endl;
}

void gpuParallelBinning(){
	

}

int main(int argc, char** argv){
	printf("num %i\n", 8);
	using namespace std;
	string fname = "s_data_02.scan";
	

	readFile(fname.c_str());
	cout<<"ShipPosition: "<<h_ship<<endl;
	cout<<"BasePosition: "<<h_baseStation<<endl;


	binInitialization();

	//cpuSequentialBinning();
	gpuParallelBinning();

	cout<<"run complete"<<endl;

	cin.get();
	return 0;

}