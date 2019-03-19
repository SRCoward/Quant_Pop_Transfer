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
void FWHT_statevec(int n, vector<complex<double>> &statevec){
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
We have now included a program to actually generate the eigenvalues in C++, as opposed to reading in from python script
We use doubles to ensure our evolution step preserves the norm.
*/
void generate_evals(int n, vector<double> &classical_evals, vector<complex<double>> &exp_classical_evals,vector<complex<double>> &exp_driver_evals, double gamma,double dt){
	int marked = n/2;
	int vec_len = pow(2,n);
	srand(time(NULL)); //initialise random seed
	vector<int> marked_left,marked_right; // these store our vectors of marked states
	double h[n]; // store the coefficients in our Hamiltonian
	double J[n][n]; // store coefficients in our Hamiltonian
	while (marked_left.size()<marked){
		// Generate pairs randomly
		int a = rand() % n; 
		int b = rand() % n;
		if (a!=b && J[a][b]!=-4){
			marked_left.push_back(a);
			marked_right.push_back(b);
			// set pair to be dimers, with strong interactions
			J[a][b] = -4;
			J[b][a] = -4;
			}
		}
	// generate random coefficients in [-1,1] for the rest of them
	for (int i=0; i<n ; i++){
		h[i] = (rand() % 2000001)*pow(10,-6) - 1; //yields random num with 6 bits of precision in [-1,1]
		for (int j=i; j<n ; j++){
			if(J[i][j]!=-4){
				J[i][j] = (rand() % 2000001)*pow(10,-6) - 1;
				J[j][i] = J[i][j];
			}
		}
	}
	// Now we actually compute the eigenvalues and exponentiate them.
	complex<double> minus_i(0,-1);
	
	for (int i=0 ; i < vec_len ; i++){
		int num_ij, num_ik, bin_ij, bin_ik;
		num_ij =i;
		double class_eval=0;
		double drive_eval=0;
		// we use the standard approach to calculating integer to binary
		for (int j=0;j<n;j++){
			bin_ij = num_ij % 2;
			num_ij/=2;
			// formula for exact eigenvalue calculations given in the report
			class_eval +=h[j]*pow(-1,bin_ij);
			drive_eval +=(abs(h[j])+1)*pow(-1,bin_ij);
			// now computing the J contributions
			num_ik=i;
			for (int k=0; k<n; k++){
				bin_ik = num_ik % 2;
				num_ik/=2;
				// formula given in the report
				class_eval+=J[j][k]*pow(-1,bin_ij+bin_ik);
				drive_eval+=(1+abs(J[j][k]))*pow(-1,bin_ij+bin_ik);
			}
		}
		// append eigenvalues and exponentiate
		classical_evals.push_back(class_eval);
		exp_classical_evals.push_back(exp(minus_i*dt*class_eval));
		exp_driver_evals.push_back(exp(minus_i*gamma*dt*drive_eval));
		}

}


/*
Will compute a single bit flip steepest descent optimisation starting from statenum and will return corresponding minima state number
*/
int steepest_descent(int n, vector<double> &classical_evals, int statenum){
	int hold_state=statenum;
	int min_state=statenum;
	int vec_len = pow(2,n);
	double old_energy = classical_evals[statenum]+1;
	double new_energy = classical_evals[statenum]; //to ensure the first loop runs
	// we continue to try bit flips until no bit flip yields a lower energy state
	while (old_energy != new_energy ){
		cout<<"old_energy="<<old_energy<<"new_energy="<<new_energy<<endl;
		old_energy = new_energy; 
		//initialise statenum to be the minimum found in the last iteration
		statenum = min_state;
		int num = statenum;
		int bin;
		// same integer to binary routine as above
		for (int i  ; i<n ; i++){
			bin = num % 2;
			num /= 2;
			// compute a bit flip
			if (bin == 0){
				hold_state = statenum + pow(2,i);
			}else{
				hold_state = statenum - pow(2,i);
			}
			// if we have found a lower energy state, update our minimum state
			if (classical_evals[hold_state]<new_energy){
				min_state = hold_state;
				new_energy = classical_evals[hold_state];
			}

		}

	}
	return min_state;
}



/*
evolve_step - implements one Trotter step of the evolution operator for the 2_local Hamiltonian. 
The program takes an input state vector of length 2^n, and the vectors of exponentiated eigenvalues for the driver and classical Hamiltonians
*/
void evolve_step(int n,vector<complex<double>> &statevec,vector<complex<double>> &exp_classical_evals,vector<complex<double>> &exp_driver_evals){
	FWHT_statevec(n,statevec);
	int vec_length = pow(2,n);
	vector<complex<double>> update(vec_length,complex<double>(0,0));
	double normalise = pow(2,-n);
	int i,k;
	//Can enable multiple threads using OMP
	//int num_threads = omp_get_num_threads();
	//double sum_threads[num_threads] = {0};
	int thread_num;
	#pragma omp parallel for private(i)
	// We compute each element of our updated vector 
	for (i=0; i<vec_length; i++){
		vector<complex<double>> classical_evec(vec_length,complex<double> (0,0));
		classical_evec[i] = 1; // apply Walsh-Hadamard transform to a computational basis state
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
	double sum=0;
	// We now copy across our updated vector into statevector. We include some checks that the operation preserves the norm
	sum=0;
	for (int i=0; i<vec_length;i++){
		statevec[i]=update[i];
		sum+=pow(real(statevec[i]),2)+pow(imag(statevec[i]),2);
	}
	sum = sqrt(sum); // this should be 1.
	cout<<"sqrt sum="<<sum<<endl;
}


int main(void){
int n;
// n controls the number of qubits in our system
n=12;
int vec_length = pow(2,n);
vector<double> classical_evals;
vector<complex<double>>exp_classical_evals;
vector<complex<double>>exp_driver_evals;

generate_evals(n,classical_evals, exp_classical_evals,exp_driver_evals,0.2,0.08);

// Randomly initialise in some state
srand(time(NULL));
int random_start_state = rand() % vec_length;
// Apply steepest-descent to locate the local minimum associated to our random start state
int starting_minima = steepest_descent(n,classical_evals,random_start_state);
vector<complex<double>> statevec(vec_length,0);
statevec.at(starting_minima)=1;

clock_t total_time = clock();
// This is the main evolution loop. We compute 50 Trotter steps, recursively.

for (int i=0;i<50;i++){
	clock_t iteration_time = clock();
	evolve_step(n,statevec,exp_classical_evals,exp_driver_evals);
	iteration_time = clock()-iteration_time;	
	cout<<i<<"iter time="<<((float)iteration_time)/CLOCKS_PER_SEC<<endl;
}

total_time = clock() - total_time;

// Write our results to results.csv which get used to plot the date
ofstream outFile("results.csv");

// We insert the energy of the starting state number at the top of our file to mark it as 'special'
double amps;
amps = pow(real(statevec[starting_minima]),2)+pow(imag(statevec[starting_minima]),2);
outFile<<classical_evals[starting_minima]<<','<<amps<<endl;

for (size_t i=0 ; i<vec_length ; i++){
	amps = pow(real(statevec[i]),2)+pow(imag(statevec[i]),2);
	outFile<<classical_evals[i]<<','<<amps<<endl;

}
outFile.close();
cout<<"starting minima "<<starting_minima<<endl;

// Output the timing data, which gets used to generate the plot in the report.
ofstream timeFile("timings.csv",ofstream::app);
timeFile<<"("<<n<<","<<((float)total_time)/CLOCKS_PER_SEC<<")"<<endl;
timeFile.close();

}
