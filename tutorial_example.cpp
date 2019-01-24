#include <QuEST.h>

#include<stdio.h>
int main(int narg, char *varg[]) {

  // load QuEST
  QuESTEnv env = createQuESTEnv();
  
  // create 2 qubits in the hadamard state
  Qureg qubits = createQureg(3, env);
  initZeroState(qubits);
	
  // apply circuit
  hadamard(qubits, 0);
  controlledNot(qubits, 0, 1);
  rotateY(qubits,2,1);
  //multiControlledPhaseGate(qubits, (int []){0, 1, 2}, 3);
  Complex a, b;
  a.real = .5; a.imag =  .5;
  b.real = .5; b.imag = -.5;
  compactUnitary(qubits, 1, a, b);

  ComplexMatrix2 u;
  u.r0c0 = (Complex) {.real=.5, .imag= .5};
  u.r0c1 = (Complex) {.real=.5, .imag=-.5}; 
  u.r1c0 = (Complex) {.real=.5, .imag=-.5};
  u.r1c1 = (Complex) {.real=.5, .imag= .5};
  
  controlledCompactUnitary(qubits, 0, 1, a, b);
  
  //multiControlledUnitary(qubits, (int []){0, 1}, 2, 2, u);
  qreal prob = getProbAmp(qubits, 7);
  printf("Probability amplitude of |111>: %lf\n", prob); 
  
  prob = calcProbOfOutcome(qubits, 2, 1);
  printf("Probability of qubit 2 being in state 1: %f\n", prob);  
  
  int outcome = measure(qubits, 0);
  printf("Qubit 0 was measured in state %d\n", outcome);
  
 outcome = measureWithStats(qubits, 2, &prob);
 printf("Qubit 2 collapsed to %d with probability %f\n", outcome, prob); 
  // unload QuEST
  destroyQureg(qubits, env); 
  destroyQuESTEnv(env);
  return 0;
}
