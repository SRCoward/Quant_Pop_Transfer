#include <QuEST.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <math.h>
#include <complex>
#include <vector>
#include <iomanip>

/*
U_gate takes three params theta, phi and lambda and returns a complex matrix
This corresponds to the U_gate described by the QASM language
*/
ComplexMatrix2 U_gate(double theta,double phi,double lambda){
  ComplexMatrix2 u;
  u.r0c0 = (Complex) {.real=cos(theta/2), .imag = 0};
  u.r0c1 = (Complex) {.real=-cos(lambda)*sin(theta/2), .imag = -sin(lambda)*sin(theta/2)};
  u.r1c0 = (Complex) {.real=cos(phi)*sin(theta/2), .imag=sin(phi)*sin(theta/2)};
  u.r1c1 = (Complex) {.real=cos(lambda+phi)*cos(theta/2), .imag= sin(lambda+phi)*cos(theta/2)};
  return u;
}


// qasm_to_quest - converts a qasm.txt input file to a QuEST circuit to be simulated.
// n represents the number of qubits
// general formula is that each line of qasm.txt represents a gate with some params. It acts on one/two qubits each
// We extract the gate type (and its params) and the qubit it acts on then add to our QuEST circuit
int qasm_to_quest(int n){
  // load QuEST
  QuESTEnv env = createQuESTEnv();

  // Create n qubits in the zero state
  Qureg qubits = createQureg(n,env);
  startRecordingQASM(qubits);
  initZeroState(qubits);

  // qasm file generated by Qiskit
  std::ifstream inFile("qasm.txt");
  std::string start_state_num;
  getline(inFile,start_state_num);
  std::vector<std::string> energies(pow(2,n));
  for (size_t j=0 ; j<pow(2,n) ; j++){
	  std::getline(inFile,energies[j]);
  }
  std::string data;
  std::string cx = "cx";
  while (!inFile.eof()){
    std::getline(inFile,data);
    //std::cout<<data<<std::endl;
    std::string op = data.substr(0,2);
    if (!op.compare("cx")){
       // If gate is CX
      char control = data[5]; //control qubit
      char target = data[10]; // target qubit
      // cast the chars to int values
      int int_control = control - '0';
      int int_target = target - '0';
      //apply controlled not gate to circuit
      controlledNot(qubits,int_control,int_target);
    } else if (!op.compare("u1")){
      // IF GATE IS U1
      std::size_t found_round = data.find(')');
      std::size_t found_square = data.find('[');
      //std::cout<<found<<std::endl;
      std::string param = data.substr(3,found_round-3);
      char target = data[found_square+1];
      int int_target = target - '0';
      //std::cout<<param<<std::endl;
      std::size_t found_e = param.find("e");
      double real_param;
      if (found_e != std::string::npos){
      	
	double signif = std::stod(param.substr(0,found_e-1));
	double exponent = std::stod(param.substr(found_e+1,found_round-found_e-1));
	//std::cout<<"signif = "<<param.substr(0,found_e-1)<<" exponent "<<param.substr(found_e+1,param.length()-found_e-1)<<std::endl;
	real_param = signif*pow(10,exponent);
      }else{ 
      	real_param = std::stod(param);
      };
      ComplexMatrix2 u = U_gate(0,0,real_param);
      //add a general unitary to the circuit
      unitary(qubits,int_target,u);

    
    } else if (!op.compare("u2")){
      //IF THE GATE IS U2
      std::size_t found_comma = data.find(',');
      std::size_t found_round = data.find(')');
      std::size_t found_square = data.find('[');
      //extract params
      std::string phi = data.substr(3,found_comma-3);
      std::string lambda = data.substr(found_comma+1, found_round - found_comma-1);
       
      char target = data[found_square+1];
      int int_target = target - '0';
      double lambda_double, phi_double;
      std::size_t found_phi_e = phi.find("e");
      std::size_t found_lambda_e = lambda.find("e");
      //check for strange values of params
      if (!phi.compare("pi")){
	phi_double = M_PI;
      } else if(found_phi_e != std::string::npos){
      	
	double signif = std::stod(phi.substr(0,found_phi_e-1));
	double exponent = std::stod(phi.substr(found_phi_e+1,phi.length()-found_phi_e-1));
	//std::cout<<"signif = "<<param.substr(0,found_e-1)<<" exponent "<<param.substr(found_e+1,found_round-found_e-1)<<std::endl;
	phi_double = signif*pow(10,exponent);
      } else{
	phi_double = std::stod(phi);
      };
      //Now compute the lambda component
      if (!lambda.compare("pi")){
	lambda_double = M_PI;
	//std::cout<<"lambda now = "<<lambda_double<<std::endl;
      }else if(found_lambda_e != std::string::npos){
	double signif = std::stod(lambda.substr(0,found_lambda_e-1));
	double exponent = std::stod(lambda.substr(found_lambda_e+1,lambda.length()-found_lambda_e-1));
	//std::cout<<"signif = "<<param.substr(0,found_e-1)<<" exponent "<<param.substr(found_e+1,found_round-found_e-1)<<std::endl;
	lambda_double = signif*pow(10,exponent);
      }else{
	lambda_double = std::stod(lambda);
	//std::cout<<"lambda now = "<<lambda_double<<std::endl;
      };
      //std::cout<<"phi"<<phi_double<<" lambda "<<lambda_double<<std::endl;
      ComplexMatrix2 u = U_gate(M_PI/2,phi_double,lambda_double);
      //now add our general unitary
      unitary(qubits,int_target,u);
    
    
    
    } else if (!op.compare("u3")){ //IF GATE IS U3
      std::size_t found_comma_1 = data.find(',');
      std::size_t found_comma_2 = data.find(',',found_comma_1+1);
      std::size_t found_round = data.find(')');
      std::size_t found_square = data.find('[');
      //std::cout<<found<<std::endl;
      std::string theta = data.substr(3,found_comma_1-3);
      std::string phi = data.substr(found_comma_1+1,found_comma_2-found_comma_1-1);
      std::string lambda = data.substr(found_comma_2+1, found_round - found_comma_2-1);
       
      std::size_t found_theta_e = theta.find("e");
      std::size_t found_phi_e = phi.find("e");
      std::size_t found_lambda_e = lambda.find("e");
      std::size_t found_theta_pi = theta.find("pi");
      std::size_t found_phi_pi = phi.find("pi");
      std::size_t found_lambda_pi = lambda.find("pi");
      //std::cout<<"theta= "<<theta<<" phi= "<<phi<<" lambda= "<<lambda<<std::endl;
      char target = data[found_square+1];
      int int_target = target - '0';
      double theta_double, lambda_double, phi_double;
      //std::cout<<"phi"<<phi<<" lambda"<<lambda<<"here"<<std::endl;
      
      if (found_theta_pi != std::string::npos){
	theta.replace(found_theta_pi,2,std::to_string(M_PI));
	theta_double = std::stod(theta);
      } else if(found_theta_e != std::string::npos){
      	
	double signif = std::stod(theta.substr(0,found_theta_e-1));
	double exponent = std::stod(theta.substr(found_theta_e+1,theta.length()-found_theta_e-1));
	//std::cout<<"signif = "<<param.substr(0,found_e-1)<<" exponent "<<param.substr(found_e+1,found_round-found_e-1)<<std::endl;
	theta_double = signif*pow(10,exponent);
      } else{
	theta_double = std::stod(theta);
      };
      if (found_phi_pi != std::string::npos){
	phi.replace(found_phi_pi,2,std::to_string(M_PI));
	phi_double = std::stod(phi);
      } else if(found_phi_e != std::string::npos){
      	
	double signif = std::stod(phi.substr(0,found_phi_e-1));
	double exponent = std::stod(phi.substr(found_phi_e+1,phi.length()-found_phi_e-1));
	//std::cout<<"signif = "<<param.substr(0,found_e-1)<<" exponent "<<param.substr(found_e+1,found_round-found_e-1)<<std::endl;
	phi_double = signif*pow(10,exponent);
      }else{
	phi_double = std::stod(phi);
      };
      if (found_lambda_pi != std::string::npos){
	lambda.replace(found_lambda_pi,2,std::to_string(M_PI));
	lambda_double = std::stod(lambda);
      }else if(found_lambda_e != std::string::npos){
	double signif = std::stod(lambda.substr(0,found_lambda_e-1));
	double exponent = std::stod(lambda.substr(found_lambda_e+1,lambda.length()-found_lambda_e-1));
	//std::cout<<"signif = "<<param.substr(0,found_e-1)<<" exponent "<<param.substr(found_e+1,found_round-found_e-1)<<std::endl;
	lambda_double = signif*pow(10,exponent);
      }else{
	lambda_double = std::stod(lambda);
      };
      ComplexMatrix2 u = U_gate(theta_double,phi_double,lambda_double);
      unitary(qubits,int_target,u);
    };
    

  };
  inFile.close();
  qreal prob = getProbAmp(qubits,0);
  stopRecordingQASM(qubits);
  // check qasm output matches
  //printRecordedQASM(qubits);
  double amp; 
  int end = pow(2,n);
  double sum = 0;
  // get amplitudes of output state, from simulating the circuit
  for (int i=0;i<end;i++){
	  amp = getProbAmp(qubits,i);
	  sum+=amp;
	  std::cout<<getProbAmp(qubits,i)<<",";
  };
  // output results to file
  std::cout<<" total prob"<<sum<<std::endl;
  std::ofstream outFile("results.csv");
  int int_start_state_num = std::stoi(start_state_num); // string to integer
  std::string round_energy = energies.at(int_start_state_num);
  round_energy = round_energy.substr(0,7);
  std::cout<<round_energy;
  outFile<<round_energy<<','<<getProbAmp(qubits,int_start_state_num)<<std::endl; 
  for(size_t j = 0; j<pow(2,n); j++){
	  std::string round_energy = energies.at(j);
	  round_energy = round_energy.substr(0,7);
	  outFile<<round_energy<<','<<getProbAmp(qubits,j)<<std::endl;
  }

  outFile.close();
  // destroy our circuit at end of execution
  destroyQureg(qubits,env);
  destroyQuESTEnv(env);

  return 0;
}

int main(void){
  int n;
  std::cout<<"input integer number of qubits"<<std::endl;
  std::cin>>n;
  qasm_to_quest(n);
}
