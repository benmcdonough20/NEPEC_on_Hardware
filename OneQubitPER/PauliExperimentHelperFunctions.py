from qiskit.providers.aer.noise import NoiseModel, kraus_error
from qiskit.quantum_info import PauliList, Pauli, Kraus
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
import numpy as np
import math
import scipy.optimize

#Symplectic inner product
def s_prod(a, b):
    if(a.compose(b) == b.compose(a)):
        return 0
    else:
        return 1
    
#The counts objects are annoyingly not dictionaries, so they don't have the has_key()
#method build in
def haskey(counts, key):
    return key in list(counts.keys())

#Get the expectation value from a counts object
def expectation(counts):
    if not haskey(counts, '0'):
        return -1
    if not haskey(counts, '1'):
        return 1
    return (counts['0']-counts['1'])/(counts['0']+counts['1'])

#return an array of the first element in an array of tuples
def unzip(zippedlist):
    return [a for (x,a) in zippedlist]

#Measure in one of the Pauli bases given an eigenvalue
def meas_basis(qc, qbit, cbit, basis):
    match basis:
        case '+':
            qc.h(qbit)
            qc.measure(qbit, cbit)
        case '-':
            qc.h(qbit)
            qc.measure(qbit, cbit)
        case 'r':
            qc.sdg(qbit)
            qc.h(qbit)
            qc.measure(qbit, cbit)
        case 'l':
            qc.sdg(qbit)
            qc.h(qbit)
            qc.measure(qbit, cbit)
        case '0':
            qc.measure(qbit,cbit)
        case '1':
            qc.measure(qbit,cbit)
        
#prepare eigenstate and generate a number of folds
def fold_state(eig, gate, folds):
    qc = QuantumCircuit(1,1)
    qc.initialize(eig)
    qc.barrier()
    for j in range(folds):
        qc.append(gate, [0])
        qc.barrier()
        qc.append(gate.inverse(), [0])
        qc.barrier()
    meas_basis(qc, 0, 0, eig)
    return qc

#Generate circuits for negative and positive eigenvalues and return as tuple
def bench_circuits(pos_eig, neg_eig, gate, folds):
    pos = fold_state(pos_eig, gate, folds)
    neg = fold_state(neg_eig, gate, folds)
    return [pos, neg]

#Reconstruct the expectation value on a Pauli operator given the expectation values
#on the positive and negative eigenstates
def reconstruct(positive, negative):
    return (positive-negative)/2

#get the expectation value on a pauli operator
def measure_fidelity(pos_circuit, neg_circuit, backend, basis_gates, noise_model, shots):
    job = execute(pos_circuit, backend, noise_model=noise_model,
                  basis_gates = basis_gates, shots = shots)
    pos = expectation(job.result().get_counts())
    job = execute(neg_circuit, backend, noise_model=noise_model, 
                  basis_gates = basis_gates, shots = shots)
    neg = expectation(job.result().get_counts())
    return reconstruct(pos, neg)

#F is a list of Paulis and T are the paulis in the model,
#coeffs is a list of lambda_k, n is the number of qubits
def build_noise_model(F, T, coeffs, n):
    omegas = []
    for (k, lambdak) in enumerate(coeffs):
        omegas.append(.5*(1+math.exp(-2*lambdak)))
        
    kraus_ops = Kraus(np.identity(2**n))
    for (P,omega,lambdak) in zip(T, omegas, coeffs):
        if lambdak != 0:
            op = Kraus([P.to_matrix()*np.sqrt(1-omega),np.sqrt(omega)*np.identity(2**n).astype(complex)])
            kraus_ops = kraus_ops.compose(op)
    #Kraus error channel
    kraus_error_channel = kraus_error(kraus_ops.data)
    kraus_noise_model = NoiseModel()
    kraus_noise_model.add_all_qubit_quantum_error(kraus_error_channel,
                                                  ['id', 'rz', 'sx', 'h', 'rx'])
    
    return kraus_noise_model

#Turn the model coefficients into fidelities
def get_ideal_fidelities(F, T, coeffs):
    M = np.zeros([len(F),len(T)])
    for (i,a) in enumerate(F):
        for (j,b) in enumerate(T):
            M[i,j] = s_prod(a,b)
    return list(zip(F, np.exp(-2*np.dot(M, coeffs))))

#generate the circuits with the folding to measure fidelities
def generate_circuits(P, gate, folds):
    pos_eig = ''
    neg_eig = ''
    
    match P:
        case 'X':
            pos_eig = '+'
            neg_eig = '-'
        case 'Y':
            pos_eig = 'r'
            neg_eig = 'l'
        case 'Z':
            pos_eig = '0'
            neg_eig = '1'
    
    circuits = []
    for i in range(folds):
        circuits.append(bench_circuits(pos_eig, neg_eig, gate, i))
        
    return circuits

#measure expectation values on a list of ciruits, corresponding to different depths
def fidelity_experiment(circuits, backend, noise_model, shots):
    fidelities = []
    for [pos_circ, neg_circ] in circuits:
        fidelities.append(measure_fidelity(pos_circ, 
                                           neg_circ, backend, 
                                           noise_model.basis_gates, noise_model, shots))

    return fidelities.copy()

#Define Walsh-hadamard transformation
def WHtransform(b, fidelities, F):
    c_b = 0
    P_b = F[b]
    for (f_a, P_a) in zip(fidelities, F):
        c_b += (-1)**s_prod(P_a, P_b)*f_a
    c_b /= len(F) #text gives normalization 2**n
    return c_b

#get the exponential fit to the data
def get_exponential_fit(fidelities):
    xrange = np.multiply(2,range(1,1+len(fidelities)))
    def Exp(t, b, a):
        return a*np.exp(-t * b)
    p0 = (.05, 1)
    params, cv = scipy.optimize.curve_fit(Exp, xrange, fidelities, p0)
    
    return params

#convert the exponential fits into fidelities
def learn_fidelities(xfidelities, yfidelities, zfidelities):
    bx, ax = get_exponential_fit(xfidelities)
    by, ay = get_exponential_fit(yfidelities)
    bz, az = get_exponential_fit(zfidelities)
    return [(Pauli('I'), 1.0), (Pauli('X'), math.exp(-bx)), (Pauli('Y'), math.exp(-by)), (Pauli('Z'), math.exp(-bz))]

#use the least-squares method to approximate the model coefficients
def learn_model_coefficients(measured_fidelities, F, T):
    measured_fidelities = unzip(measured_fidelities)
    
    M = np.zeros([len(F),len(T)])
    for (i,a) in enumerate(F):
        for (j,b) in enumerate(T):
            M[i,j] = s_prod(a,b)
            
    def lsq_fit(coeffs):
        return np.sum(np.dot(M,coeffs)+np.log(measured_fidelities)/2)

    coeffs_guess = scipy.optimize.nnls(M, -.5*np.log(measured_fidelities))
    measured_coeffs = coeffs_guess[0]
    
    return list(zip(T,measured_coeffs))