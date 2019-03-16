#include<iostream>
#include<vector>
#include<complex>
#include<math.h>
#include<fstream>
#include<string>
#include<iomanip>
#include<omp.h>
using namespace std;

/*
The program computes the evolution of our initial state under action of the 2_local Hamiltonian
This is aims to mimic the circuit simulation by maintaining exponential state vectors.
*/

//FWHT_statevec computes the Fast Walsh-Hadamard transform of the input statevec, which is of length 2^n
// Code based off the example given at: https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
void FFT_statevec(int n, vector<complex<double>> &statevec){
	int layer = 1;
	double normalise = sqrt(pow(2,-n));
	while (layer<pow(2,n)){
		for (int i=0;i<pow(2,n);i+=layer*2){
			for (int j=i;j<i+layer;j++){
				complex<double> x = statevec[j];
				complex<double> y = statevec[j + layer];
				statevec[j] = x + y;
				statevec[j+layer] = x - y;
				if (layer >= pow(2,n-1)-0.01){
					statevec[j]*=normalise;
					statevec[j+layer]*=normalise;
				}
			}
		}
		layer*=2;
	}
}

//We read in the exponentiated eigenvalues of the classical and driver components of our 2_local Hamiltonian, as described in the report
void read_in_evals(int n,vector<complex<double>> &exp_classical_evals,vector<complex<double>> &exp_driver_evals){
	ifstream inFile("exp_classical_evals.txt");
	string line;
	int i = 0;
	while (i<pow(2,n)){
		getline(inFile,line);
		size_t comma = line.find(',');
		double a = std::stod(line.substr(0,comma-1));
		double b = std::stod(line.substr(comma+1)); 
		complex<double> ab = {a,b};	
		exp_classical_evals.push_back(ab);
		 		
		i+=1;
//		cout<<"a="<<complex<double>(a,b)<<"  "<<exp_classical_evals.at(i-1)<<endl;
	}
	inFile.close();
	ifstream inFile2("exp_driver_evals.txt");
	i=0;
	while(i<pow(2,n)){
		getline(inFile2,line);
		size_t comma = line.find(',');
		double a = std::stod(line.substr(0,comma-1));
		double b = std::stod(line.substr(comma+1)); 
		
		exp_driver_evals.push_back(complex<double>(a,b));
		i+=1;
	}
	inFile2.close();
}

/*
evolve_step - implements one Trotter step of the evolution operator for the 2_local Hamiltonian.
The program takes an input state vector of length 2^n, and the vectors of exponentiated eigenvalues for the driver and classical Hamiltonians
*/
void evolve_step(int n,vector<complex<double>> &statevec,vector<complex<double>> &exp_classical_evals,vector<complex<double>> &exp_driver_evals){
	FFT_statevec(n,statevec);
	int vec_length = pow(2,n);
	vector<complex<double>> update(vec_length,complex<double>(0,0));
	double normalise = pow(2,-n);
	int i,k;
	int num_threads = omp_get_num_threads();
	double sum_threads[num_threads] = {0};
	int thread_num;
	#pragma omp parallel for private(i)
	for (i=0; i<vec_length; i++){
		vector<complex<double>> classical_evec(vec_length,complex<double> (0,0));
		classical_evec[i] = 1;
		FFT_statevec(n,classical_evec);
		for (k=0;k<vec_length;k++){
			update[i]+=statevec[k]*exp_driver_evals[k]*classical_evec[k];
		}
		update[i]*= exp_classical_evals[i];
		if (real(update[i])>pow(10,11)){
				cout<<"update value "<<i<<" "<<update[i]<<endl;
				cout<<"classical_evec "<<classical_evec[i]<<endl;
				cout<<"driver_evals "<<exp_driver_evals[i]<<endl;
		}
	// enable multithreading
	//thread_num = omp_get_thread_num();
	//sum_threads[thread_num]+=pow(real(update[i]),2)+pow(imag(update[i]),2);
	}
	double sum=0;
    // We added some checking code to ensure that our operator acts as a norm preserving operation, we can exclude this to improve runtimes
	sum=0;
	for (int i=0; i<vec_length;i++){
		sum+=pow(real(update[i]),2)+pow(imag(update[i]),2);
	}
	sum = sqrt(sum);	
	cout<<"sqrt sum="<<sum<<endl;
	for (int i=0; i<vec_length; i++){
		statevec[i] = update[i]*pow(sum,-1);
	}
}


int main(void){
int n,start_state_num;
// n controls the number of qubits in our system
// start_state_num - corresponds to the number of the starting state which gets represented in binary
cout<<"input integer n:"<<endl;
n=13;
int vec_length = pow(2,n);
// we read in the classical energies
ifstream inFile("classical_evals.txt");
vector<string> energies(vec_length);
for (size_t j=0; j<vec_length; j++){
	getline(inFile,energies[j]);
}
inFile.close();
cout<<"input start state num:"<<endl;
cin>>start_state_num;

vector<complex<double>> statevec(vec_length,0);
statevec.at(start_state_num)=1;

vector<complex<double>>exp_classical_evals;
vector<complex<double>>exp_driver_evals;
read_in_evals(n,exp_classical_evals,exp_driver_evals);

// This is the main evolution loop. We compute 50 Trotter steps, recursively.
for (int i=0;i<50;i++){
	cout<<i<<endl;
	evolve_step(n,statevec,exp_classical_evals,exp_driver_evals);
};
// Write our results to results.csv which get used to plot the date
ofstream outFile("results.csv");
for (size_t i=0 ; i<vec_length ; i++){
	string round_energy = energies.at(i);
	round_energy = round_energy.substr(0,7);
	double amps;
	amps = pow(real(statevec[i]),2)+pow(imag(statevec[i]),2);
	outFile<<round_energy<<','<<amps<<endl;

}
outFile.close();
}
