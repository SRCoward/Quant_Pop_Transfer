#include<iostream>
#include<vector>
#include<complex>
#include<math.h>
#include<fstream>
#include<string>
#include<iomanip>
#include<omp.h>
#include<time.h>
#include<stdlib.h>
#include<cstdlib>
using namespace std;

/*
The program computes the evolution of our initial state under action of the 2_local Hamiltonian
This is aims to mimic the circuit simulation by maintaining exponential state vectors.
*/

//FWHT_statevec computes the Fast Walsh-Hadamard transform of the input statevec, which is of length 2^n
// Code based off the example given at: https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
void FWHT_statevec(int n, vector<complex<long double>> &statevec){
	int layer = 1;
	double normalise = sqrt(pow(2,-n));
	while (layer<pow(2,n)){
		for (int i=0;i<pow(2,n);i+=layer*2){
			for (int j=i;j<i+layer;j++){
				complex<long double> x = statevec[j];
				complex<long double> y = statevec[j + layer];
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
void read_in_evals(int n,vector<complex<long double>> &exp_classical_evals,vector<complex<long double>> &exp_driver_evals){
	ifstream inFile("exp_classical_evals.txt");
	string line;
	int i = 0;
	
	while (i<pow(2,n)){
		getline(inFile,line);
		size_t comma = line.find(',');	
		long double a = std::stod(line.substr(0,comma-1));
		long double b = std::stod(line.substr(comma+1)); 
		complex<long double> ab = {a,b};	
		exp_classical_evals.push_back(ab);	
		i+=1;
	}

	inFile.close();
	ifstream inFile2("exp_driver_evals.txt");
	i=0;

	while(i<pow(2,n)){
		getline(inFile2,line);
		size_t comma = line.find(',');	
		long double a = std::stod(line.substr(0,comma-1));
		long double b = std::stod(line.substr(comma+1)); 
		
		exp_driver_evals.push_back(complex<long double>(a,b));
		i+=1;
	}
	inFile2.close();
}

void generate_evals(int n, vector<complex<long double>> &exp_classical_evals,vector<complex<long double>> &exp_driver_evals, long double gamma,long double dt){
	int marked = n/2;
	int vec_len = pow(2,n);
	srand(time(NULL)); //initialise random seed
	vector<int> marked_left,marked_right;
	double h[n];
	double J[n][n];
	while (marked_left.size()<marked){
		int a = rand() % n;
		int b = rand() % n;
		if (a!=b && J[a][b]!=-4){
			marked_left.push_back(a);
			marked_right.push_back(b);
			J[a][b] = -4;
			J[b][a] = -4;
			}
		}
	for (int i=0; i<n ; i++){
		h[i] = (rand() % 2000001)*pow(10,-6) - 1; //yields random num with 6 bits of precision in [-1,1]
		//cout<<h[i]<<endl;
		for (int j=i; j<n ; j++){
			if(J[i][j]!=-4){
				J[i][j] = (rand() % 2000001)*pow(10,-6) - 1;
				J[j][i] = J[i][j];
				//cout<<'('<<i<<','<<j<<')'<<J[i][j]<<endl;
			}
		}
	}
	// Now we actually compute the eigenvalues...
	complex<long double> minus_i(0,-1);
	
	for (int i=0 ; i < vec_len ; i++){
		int num_ij, num_ik, bin_ij, bin_ik;
		num_ij =i;
		long double class_eval=0;
		long double drive_eval=0;
		for (int j=0;j<n;j++){
			bin_ij = num_ij % 2;
			//cout<<bin_ij;
			num_ij/=2;
			class_eval +=h[j]*pow(-1,bin_ij);
			drive_eval +=(abs(h[j])+1)*pow(-1,bin_ij);
			num_ik=i;
			for (int k=0; k<n; k++){
				bin_ik = num_ik % 2;
				num_ik/=2;
				class_eval+=J[j][k]*pow(-1,bin_ij+bin_ik);
				drive_eval+=(1+abs(J[j][k]))*pow(-1,bin_ij+bin_ik);
			}
		}
		exp_classical_evals.push_back(exp(minus_i*dt*class_eval));
		exp_driver_evals.push_back(exp(minus_i*gamma*dt*drive_eval));
		}

}


/*
evolve_step - implements one Trotter step of the evolution operator for the 2_local Hamiltonian. 
The program takes an input state vector of length 2^n, and the vectors of exponentiated eigenvalues for the driver and classical Hamiltonians
*/
void evolve_step(int n,vector<complex<long double>> &statevec,vector<complex<long double>> &exp_classical_evals,vector<complex<long double>> &exp_driver_evals){
	FWHT_statevec(n,statevec);
	int vec_length = pow(2,n);
	vector<complex<long double>> update(vec_length,complex<long double>(0,0));
	double normalise = pow(2,-n);
	int i,k;
	//Can enable multiple threads using OMP
	//int num_threads = omp_get_num_threads();
	//double sum_threads[num_threads] = {0};
	int thread_num;
	#pragma omp parallel for private(i)
	// We compute each element of our updated vector 
	for (i=0; i<vec_length; i++){
		vector<complex<long double>> classical_evec(vec_length,complex<long double> (0,0));
		classical_evec[i] = 1;
		FWHT_statevec(n,classical_evec);
		for (k=0;k<vec_length;k++){
			update[i]+=statevec[k]*exp_driver_evals[k]*classical_evec[k];
		}
		update[i]*= exp_classical_evals[i];
		/*
		cout<<"update value "<<i<<" "<<update[i]<<endl;
		cout<<" classical_evec= "<<classical_evec[i]<<endl;
		cout<<" driver_evals= "<<exp_driver_evals[i]<<endl;
		*/
	}
	long double sum=0;
	// We added some checking code to ensure that our operator acts as a norm preserving operation, we can exclude this to improve runtimes
	sum=0;
	for (int i=0; i<vec_length;i++){
		sum+=pow(real(update[i]),2)+pow(imag(update[i]),2);
	}
	sum = sqrt(sum);	
	cout<<"sqrt sum="<<sum<<endl;
	for (int i=0; i<vec_length; i++){
		// We implement a bit of a hack to renormalise our updated vector
		statevec[i] = update[i]*pow(sum,-1);
	}
}


int main(void){
int n,start_state_num;
// n controls the number of qubits in our system
// start_state_num - corresponds to the number of the starting state which gets represented in binary
cout<<"input integer n:"<<endl;
n=15;
int vec_length = pow(2,n);

ifstream inFile("classical_evals.txt");
vector<string> energies(vec_length);
// we read in the classical energies
for (size_t j=0; j<vec_length; j++){
	getline(inFile,energies[j]);
}
inFile.close();
cout<<"input start state num:"<<endl;
cin>>start_state_num; // we input the start_state_num at runtime

vector<complex<long double>> statevec(vec_length,0);
statevec.at(start_state_num)=1;

vector<complex<long double>>exp_classical_evals;
vector<complex<long double>>exp_driver_evals;
generate_evals(n,exp_classical_evals,exp_driver_evals,0.2,0.08);
//read_in_evals(n,exp_classical_evals,exp_driver_evals);
clock_t total_time = clock();
// This is the main evolution loop. We compute 50 Trotter steps, recursively.

for (int i=0;i<50;i++){
	cout<<i<<endl;
	evolve_step(n,statevec,exp_classical_evals,exp_driver_evals);
	}

total_time = clock() - total_time;

// Write our results to results.csv which get used to plot the date
ofstream outFile("results.csv");
string round_energy = energies.at(start_state_num);
round_energy = round_energy.substr(0,7); // we truncate the energies
// We insert the energy of the starting state number at the top of our file to mark it as 'special'
double amps;
amps = pow(real(statevec[start_state_num]),2)+pow(imag(statevec[start_state_num]),2);
outFile<<round_energy<<','<<amps<<endl;

for (size_t i=0 ; i<vec_length ; i++){
	round_energy = energies.at(i);
	round_energy = round_energy.substr(0,7);
	amps = pow(real(statevec[i]),2)+pow(imag(statevec[i]),2);
	outFile<<round_energy<<','<<amps<<endl;

}
outFile.close();

// Output the timing data, which gets used to generate the plot in the report.
ofstream timeFile("timings.csv",ofstream::app);
timeFile<<"("<<n<<","<<((float)total_time)/CLOCKS_PER_SEC<<")"<<endl;
timeFile.close();

}
