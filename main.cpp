#include <stdio.h>
#include <stdlib.h> 
#include <iostream>
#include <fstream>
#include <string>

struct Asteroid{
	float x, y, value;
};

struct Point{
	float x, y;
};




Asteroid * asteroids;
Point ship;
Point baseStation;

unsigned long long numAsteroids;

Asteroid readFile(const char* filename){
	using namespace std;

	cout<<"in readfile"<<endl;

	ifstream inFile(filename, ios::in | ios::binary);
	char line [sizeof(float)*12];//3*sizeof(float)]; 
	//string s = "";
	if(inFile.is_open()){
		inFile.seekg (0, ios::beg);
		//read base station position
		float f;
		inFile.read(reinterpret_cast<char*>(&baseStation.x), sizeof(float));
		inFile.read(reinterpret_cast<char*>(&baseStation.y), sizeof(float));
		inFile.read(reinterpret_cast<char*>(&f), sizeof(float));

		int pos = inFile.tellg ();		

		inFile.seekg (0, ios::end);
   		numAsteroids = (unsigned long long)(((int)inFile.tellg())-pos)/12;   		
    	inFile.seekg (pos, ios::beg);


		asteroids = new Asteroid [numAsteroids*sizeof(Asteroid)];

		int count;

		while(inFile.good()){	

			
			inFile.read(reinterpret_cast<char*>(&f), sizeof(f));

			//cout<<"Float val: "<<f<<" "<<2*f<<endl;


		}
	}

	inFile.close();

}












int main(int argc, char** argv){
	printf("num %i\n", 8);
	using namespace std;
	string fname = "s_data_02.scan";
	

	readFile(fname.c_str());
	


}