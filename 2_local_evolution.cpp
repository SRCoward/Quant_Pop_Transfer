#include<iostream>
#include<vector>
#include<complex>
#include<math.h>
#include<fstream>
#include<string>
#include<iomanip>
using namespace std;
//we generate a subset of size roughly n/2 of subsets of bonds
/*void generate_marked_bonds(int n,std::vector<std::tuple<int,int>> marked_bonds){
	int max_bonds = 0.5*n*(n-1);
	std::vector<std::tuple<int,int> > all_bonds;
	int num_marked_bonds = n/2
	for (int i = 0; i<n ; i++){
		for (int j = i ; j<n ; j++){
			all_bonds.push_back(std::make_tuple(i,j));
			
		}
	}
	std::cout<<all_bonds<<std::endl;
	for (int i=0; i<num_marked_bonds; i++){
	
	}
		
	
		
}
*/
//Lets just read in the useful stuff that we actually need as have already generated all the data in python
void FFT_statevec(int n, complex<double> statevec[]){
	int layer = 1;
	while (layer<pow(2,n)){
		for (int i=0;i<pow(2,n);i+=layer*2){
			for (int j=i;j<i+layer;j++){
				complex<double> x = statevec[j];
				complex<double> y = statevec[j + layer];
				statevec[j] = x + y;
				statevec[j+layer] = x - y;
			}
		}
		layer*=2;
	}
//	int power = n;	
//	double divisor = sqrt(1/pow(2,power));
//	complex& operator*=(statevec,1);
}
void read_in_evals(complex<double>exp_classical_evals[],complex<double>exp_driver_evals[]){
	ifstream inFile("exp_classical_evals.txt");
	string line;
	int i = 0;
	while (!inFile.eof()){
		getline(inFile,line);
		size_t comma = line.find(',');
		double a =  std::stod(line.substr(0,comma-1));
		double b = std::stod(line.substr(comma+1)); 
	//	double b;
		exp_classical_evals[i] = (a,b);
		 		
		i+=1;
		cout<<exp_classical_evals[i]<<endl;
	}
	inFile.close();
	ifstream inFile2("exp_driver_evals.txt");
	i=0;
	while(!inFile2.eof()){
		getline(inFile2,line);
		size_t comma = line.find(',');
		double a = std::stod(line.substr(0,comma-1));
		double b = std::stod(line.substr(comma+1)); 
		exp_driver_evals[i] = (a,b);
		//real(exp_driver_evals[i]) = stod(line.substr(0,comma-1)); 
		//imag(exp_driver_evals[i]) = stod(line.substr(comma+1,line.end()));
		i+=1;
		cout<<exp_driver_evals[i]<<endl;
	}
	inFile2.close();
}


void evolve_step(int n,complex<double>statevec[],double dt[],double classical_evals[],double driver_evals[], double gamma){
	FFT_statevec(n,statevec);
	int vec_length = pow(2,n);
	complex<double> classical_evec[vec_length] = 0;
}


int main(void){
//do something useful here
int marked_bonds[1][2];
//generate_marked_bonds(2,marked_bonds);
complex<double> statevec[8] = {0,1,0,0,0,0,0,0};
FFT_statevec(3,statevec);
for (int i=0;i<8;i++){
	cout<<statevec[i]<<endl;
}
//TO DO:  FOR the statevector it is not normalised so be careful will need to include this factor back in at some stage

}
