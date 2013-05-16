#include <stdio.h>
#include <stdlib.h> 
#include <iostream>
#include <fstream>
#include <string>

void readFile(const char* filename){
	using namespace std;

	ifstream inFile(filename, ios::in | ios::binary);
	char line [sizeof(float)];//3*sizeof(float)]; 
	//string s = "";
	if(inFile.is_open()){
		inFile.seekg (0, ios::beg);
		while(inFile.good()){

		inFile.read(line, sizeof(float));
		//float d1 = atof(s.c_str());
		//printf("%s\n", s);
		cout<<atof(line)<<endl;
		




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