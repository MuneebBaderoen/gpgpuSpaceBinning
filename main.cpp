#include <stdio.h>
#include <stdlib.h> 
#include <iostream>
#include <fstream>
#include <string>


char * word;


void readFile(const char* filename){
	using namespace std;

	cout<<"in readfile"<<endl;

	ifstream inFile(filename, ios::in | ios::binary);
	char line [sizeof(float)*8];//3*sizeof(float)]; 
	//string s = "";
	if(inFile.is_open()){
		inFile.seekg (0, ios::beg);
		while(inFile.good()){	

			float f;
			inFile.read(reinterpret_cast<char*>(&f), sizeof(f));

			cout<<"Float val: "<<f<<" "<<2*f<<endl;


		}
	}

	inFile.close();

}












int main(int argc, char** argv){
	printf("num %i\n", 8);
	using namespace std;
	string fname = "s_data_02.scan";
	// /cout<<"Word: "<<word<<endl;

	readFile(fname.c_str());
	//cout<<"Word2: "<<word<<endl;


}