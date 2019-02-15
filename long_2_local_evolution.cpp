#include<iostream>
#include<vector>
#include<complex>
#include<math.h>
#include<fstream>
#include<string>
#include<iomanip>
#include<omp.h>
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
void FFT_statevec(int n, vector<complex<long double>> &statevec){
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
//	int power = n;	
//	double divisor = sqrt(1/pow(2,power));
//	complex& operator*=(statevec,1);
}
void read_in_evals(int n,vector<complex<long double>> &exp_classical_evals,vector<complex<long double>> &exp_driver_evals){
	ifstream inFile("exp_classical_evals.txt");
	string line;
	int i = 0;
	while (i<pow(2,n)){
		getline(inFile,line);
		size_t comma = line.find(',');
//		cout<<line.substr(0,comma-1)<<line.substr(comma+1)<<endl;	
		long double a = std::stod(line.substr(0,comma-1));
		long double b = std::stod(line.substr(comma+1)); 
		complex<long double> ab = {a,b};	
		exp_classical_evals.push_back(ab);
		 		
		i+=1;

		//cout<<"a="<<complex<double>(a,b*10)<<"  "<<exp_classical_evals.at(i-1)<<endl;

	}
	inFile.close();
	ifstream inFile2("exp_driver_evals.txt");
	i=0;
	while(i<pow(2,n)){
		getline(inFile2,line);
		size_t comma = line.find(',');
//		cout<<line.substr(0,comma-1)<<line.substr(comma+1)<<endl;	
		long double a = std::stod(line.substr(0,comma-1));
		long double b = std::stod(line.substr(comma+1)); 
		
		exp_driver_evals.push_back(complex<long double>(a,b));
		//real(exp_driver_evals[i]) = stod(line.substr(0,comma-1)); 
		//imag(exp_driver_evals[i]) = stod(line.substr(comma+1,line.end()));
		i+=1;
		//cout<<exp_driver_evals[i-1]<<endl;
//		cout<<"a="<<complex<double>(a,b)<<endl;
	}
	inFile2.close();
}


void evolve_step(int n,vector<complex<long double>> &statevec,vector<complex<long double>> &exp_classical_evals,vector<complex<long double>> &exp_driver_evals){
	FFT_statevec(n,statevec);
	int vec_length = pow(2,n);
	vector<complex<long double>> update(vec_length,complex<long double>(0,0));
	double normalise = pow(2,-n);
	int i,k;
	//int num_threads = omp_get_num_threads();
	//double sum_threads[num_threads] = {0};
	int thread_num;
	#pragma omp parallel for private(i)
	for (i=0; i<vec_length; i++){
		vector<complex<long double>> classical_evec(vec_length,complex<long double> (0,0));
		classical_evec[i] = 1;
		FFT_statevec(n,classical_evec);
		for (k=0;k<vec_length;k++){
			update[i]+=statevec[k]*exp_driver_evals[k]*classical_evec[k];
		}
		update[i]*= exp_classical_evals[i];
		//cout<<update[i];
		/*
		cout<<"update value "<<i<<" "<<update[i]<<endl;
		cout<<" classical_evec= "<<classical_evec[i]<<endl;
		cout<<" driver_evals= "<<exp_driver_evals[i]<<endl;
		*/
	}
	long double sum=0;
	/*
	for (int j=0;j<num_threads;j++){
		sum+=sum_threads[j];
	}
	cout<<"sum is ="<<sum<<endl;
	*/
	sum=0;
	for (int i=0; i<vec_length;i++){
		sum+=pow(real(update[i]),2)+pow(imag(update[i]),2);
	}
	sum = sqrt(sum);	
	cout<<"sqrt sum="<<sum<<endl;
	for (int i=0; i<vec_length; i++){
		statevec[i] = update[i]*pow(sum,-1);
		//sum+=pow(real(update[i]),2)+pow(imag(update[i]),2);
	}
}


int main(void){
int n,start_state_num;
cout<<"input integer n:"<<endl;
n=13;
int vec_length = pow(2,n);
ifstream inFile("classical_evals.txt");
vector<string> energies(vec_length);
for (size_t j=0; j<vec_length; j++){
	getline(inFile,energies[j]);
}
inFile.close();
cout<<"input start state num:"<<endl;
cin>>start_state_num;

vector<complex<long double>> statevec(vec_length,0);
statevec.at(start_state_num)=1;

vector<complex<long double>>exp_classical_evals;
vector<complex<long double>>exp_driver_evals;
//exp_driver_evals[0]=complex<double>(1,0);
//cout<<exp_classical_evals[0]<<"hello!"<<endl;
read_in_evals(n,exp_classical_evals,exp_driver_evals);
//TO DO:  FOR the statevector it is not normalised so be careful will need to include this factor back in at some stage
//exp_classical_evals[0]=complex<double>(1,0);
for (int i=0;i<50;i++){
	cout<<i<<endl;
	evolve_step(n,statevec,exp_classical_evals,exp_driver_evals);
};

ofstream outFile("results.csv");
string round_energy = energies.at(start_state_num);
round_energy = round_energy.substr(0,7);
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
}