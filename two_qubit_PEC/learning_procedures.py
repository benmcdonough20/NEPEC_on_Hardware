'''
This package provides an implementation of a fidelity learning procedure for single qubit,
along with some extra visualization and debugging tools.

The structure is designed like the qiskit_experiments package. There are two classes:

    1) The experiment class
    2) The analysis class

The experiment class is used to generate the circuits necessary to benchmark a layer of single-
qubit gates. The class implements the following functionalities

    - Use backend coupling map to generate the nine pauli measurements (Following fig. S3)
    - Construct a gate dictionary to compile circuits with uniform noise
    - Generate single instance of twirled circuits with corresponding metadata (Following fig. S1)

        * Implement basis change
        * Add k layers of pauli twirling
        * Change to computational basis
        * Twirl readout in Z basis
        * Compile adjacent gates such that

            a) SPAM errors are minimized
            b) Noise on gates is uniform 

    - Generate all circuits required to benchmark layer with corresponding metadata

The analysis class takes in counts from qiskit.providers.ibmq.job.result().get_counts() and
the associated metadata to find the fidelities and generate the model coefficients. The class
is a counterpart to the experiment class and can be initialized from an instance of the
experiment class. This class implements the following features:
    
    - extracting the error model from the backend and computing twirled channel fidelities
    - generating a list of paulis in the sparse model bases on backend topology
    - converting single shot into expectation value using readout twirling and pauli metadata
    - processing experiment data to create fidelity list and compute model coefficients
    - generate graphs of exponential fits vs circuit depth
    - create graphs of measured fidelities vs predicted fidelities
    - show graph of measured model coefficients vs predicted coefficients

'''

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli, pauli_basis, SuperOp, PTM
from random import choice, choices
from itertools import product, permutations, cycle
from scipy.optimize import curve_fit, nnls
from matplotlib import pyplot as plt
import numpy as np
from qiskit.providers.aer.noise import NoiseModel

#display a progress bar for my sanity
def progressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()

#generate circuits and metadata for learning the noise in a layer of single-qubit gates
class SingleGateLayerExperiment:

    '''
    layer: a list of qubit with gates contained in the layer being benchmarked
    backend: IBMQBackend to be run on 
    samples: samples from twirl to take
    depths: noise repetitions to use for exponential fit
    '''
    def __init__(self, layer, backend, samples=64, depths = [2,4,8,16,32,64,128]):
        #initialize variables
        self.layer = layer
        self.depths = depths
        self.backend = backend
        self.samples = samples

        self.transpiled_gates = {} #store dictionary of transpiled gates
        self.raw_gates = {} #store dictionary of pauli+H+S gates for viewing
        self.n = len(layer) #number of qubits nontrivially affected in layer
        self.num_qubits = len(backend.properties().to_dict()["qubits"])

        #layer must have less qubits than backend
        if self.n > self.num_qubits:
            raise Exception("Layer cannot fit on backend")

        #parse coupling map to contain only qubits in layer
        proc_coupling = backend.configuration().coupling_map
        self.coupling_map = [(q1, q2) for q1,q2 in proc_coupling if q1 in layer and q2 in layer]

        #generate adjacency matrix, pauli strings, and transpilation dictionaries for later use
        self.adjacency_matrix = self.get_connectivity(backend)
        self.pauli_strings = self.get_pauli_strings()
        self.raw_gates, self.transpiled_gates = self._generate_gate_dict(backend)

    #parse coupling map to generate adjacency matrix and find the degree of each
    def get_connectivity(self):
        qubits = self.layer
        connections = self.coupling_map.copy() 
        n = self.n

        #remap qubits so that layer is sequential
        unmapped = lambda i: qubits.index(i)
        verts = [unmapped(qubit) for qubit in qubits]
        edges = [(unmapped(qubit1), unmapped(qubit2)) for qubit1,qubit2 in connections]

        #adjacency matrix has a 1 at i,j if i and j are connected, 0 otherwise
        adjacency_matrix = [[0 for i in verts] for j in verts] 
        for (vert1,vert2) in edges:
            adjacency_matrix[vert1][vert2] = 1
            adjacency_matrix[vert2][vert1] = 1

        return adjacency_matrix

    #The method acts as a head for the recursive sweeping procedure used to generate the pauli strings
    def get_pauli_strings(self):
        verts = range(self.n) #qubits in layer remapped in numerical order
        start_vertex = 0 #this should be chosen to be an edge qubit
        self.pauli_strings = [['I']*self.n for i in range(9)]
        visited_verts = [] #keep track of veritices for which the bases are already selected
        while(True): #if there are isolated regions _getstr bottoms out, this method restarts it
            self._getstr(start_vertex, visited_verts)
            remaining_verts = [v for v in verts if v not in visited_verts]
            if len(remaining_verts) == 0:
                self.pauli_strings = ["".join(arr) for arr in self.pauli_strings]
                return self.pauli_strings 
            else: 
                self._getstr(remaining_verts[0], visited_verts)

    #recursive sweeping procedure to find pauli strings based on number of predecessor vertices
    def _getstr(self, vertex, visited_verts):
        pauli_strings = self.pauli_strings
        adjacency_matrix = self.adjacency_matrix
        
        #copied from Fig. S3 in van den Berg
        example_orderings = {"XXXYYYZZZ":"XYZXYZXYZ",
                            "XXXYYZZZY":"XYZXYZXYZ",
                            "XXYYYZZZX":"XYZXYZXYZ",
                            "XXZYYZXYZ":"XYZXZYZYX",
                            "XYZXYZXYZ":"XYZZXYYZX"}

        visited_verts.append(vertex)
        children = [i for i,e in enumerate(adjacency_matrix[vertex]) if e == 1]
        visited_children = [c for c in children if c in visited_verts]

        match len(visited_children):
            case 0:
                cycp = cycle("XYZ")
                for i,s in enumerate(pauli_strings):
                    pauli_strings[i][vertex] = next(cycp)

            case 1:
                predecessor = visited_children[0]
                #store permutation of indices so that predecessor has X,X,X,Y,Y,Y,Z,Z,Z
                reorder_list = [[] for i in range(3)]
                for i in range(9):
                    basis = pauli_strings[i][predecessor]
                    reorder_list["XYZ".index(basis)].append(i)
                
                for i in range(3):
                    for j,c in enumerate("XYZ"):
                        idx = reorder_list[i][j]
                        pauli_strings[idx][vertex] = c

            case 2:
                predecessor0 = visited_children[0]
                predecessor1 = visited_children[1]

                #use the same reordering trick to get XXXYYYZZZ on first predecessor
                reorder_list = [[] for i in range(3)] 
                for i in range(9):
                    basis = pauli_strings[i][predecessor0]
                    reorder_list["XYZ".index(basis)].append(i)
                
                #list out string with permuted values of predecessor 2
                substring = ""
                for list in reorder_list:
                    for idx in list:
                        substring += pauli_strings[idx][predecessor1]

                #match predecessor two with a permutation of example_orderings
                reordering = ""
                for perm in permutations("XYZ"):
                    p_string = "".join(["XYZ"[perm.index(p)] for p in substring])
                    if p_string in example_orderings:
                        reordering = example_orderings[p_string]
                        break
                
                #unpermute the example orderings so that they match the original strings
                i = 0
                for list in reorder_list:
                    for idx in list:
                        pauli_strings[idx][vertex] = reordering[i]
                        i += 1

            case _: #processor needs to have connectivity so that there are <= 2 predecessors
                raise Exception("Three or more predecessors encountered")
        
        for c in children: #call recursive method on children
            if c not in visited_children:
                self._getstr(c, visited_verts)

    def _generate_gate_dict(self, backend):
        transpiled_gates = {}
        raw_gates = {}

        for p in "IXYZ":
            qc = QuantumCircuit(1)
            qc.append(Pauli(p),[0])
            transpiled_gates[p] = transpile(qc, basis_gates=backend._basis_gates())
            raw_gates[p] = qc

        for h,p in product(["H","HS"],"IXYZ"):
            qc = QuantumCircuit(1)
            match h:
                case "H":
                    qc.h(0)
                case "HS":
                    qc.h(0)
                    qc.s(0)
                case "I":
                    qc.id(0)
            qc.append(Pauli(p),[0])
            transpiled_gates[h+p] = transpile(qc, basis_gates = backend._basis_gates())
            raw_gates[h+p] = qc

        for p,h,x in product("IXYZ", ["", "H", "SdgH"], ["","X"]):
            qc = QuantumCircuit(1)
            qc.append(Pauli(p),[0])
            match h:
                case "H":
                    qc.h(0)
                case "SdgH":
                    qc.sdg(0)
                    qc.h(0)
            if x=="X":
                qc.x(0)

            transpiled_gates[p+h+x] = transpile(qc, basis_gates = backend._basis_gates())
            raw_gates[p+h+x] = qc      

        transpiled_gates['I'].id(0)
        transpiled_gates['Z'].id(0)
        transpiled_gates['XX'].id(0)

        return raw_gates, transpiled_gates

    def print_gate_conversions(self):
        raw_gates = self.raw_gates
        transpiled_gates = self.transpiled_gates

        qc = QuantumCircuit(len(raw_gates))
        for i,p in enumerate(raw_gates.keys()):
            qc = qc.compose(raw_gates[p], [i])


            qc2 = QuantumCircuit(len(transpiled_gates))

        for i,p in enumerate(transpiled_gates.keys()):
            qc2 = qc2.compose(transpiled_gates[p], [i])

        qc.barrier()
        qc = qc.compose(qc2)
        return qc
    
    #generate pauli measurement circuitry with corresponding metadata
    def generate_single_instance(self, basis_operator, noise_repetitions, transpiled = True):
        n = self.n
        if transpiled:
            gates = self.transpiled_gates
            qubits = self.layer
            qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        else:
            gates = self.raw_gates
            qubits = range(n)
            qc = QuantumCircuit(n)

        #choose first layer
        twirl_layer = [Pauli(p) for p in choices("IXYZ",k=n)]
        #Choose B gates
        for i,(b,p) in enumerate(zip(basis_operator, twirl_layer)):
            gate_name = {'X':"H", 'Y':"HS"}.get(b,"")
            gate_name += p.to_label()
            qc = qc.compose(gates[gate_name], [qubits[i]])

        qc.barrier()
        #add the twirled layers
        for l in range(noise_repetitions-1):
            for i in range(n):
                tw_op = choice("IXYZ")
                qc = qc.compose(gates[tw_op], [qubits[i]])
                twirl_layer[i] = twirl_layer[i].compose(Pauli(tw_op))
            qc.barrier()

        binstr = choices('01',k=n)

        for i,(p,b,x) in enumerate(zip(twirl_layer, basis_operator, binstr)):
            gate_name = p.to_label()[-1]
            gate_name += {'X':'H', 'Y':"SdgH"}.get(b,"")
            gate_name += {'0':'', '1':'X'}[x]
            qc = qc.compose(gates[gate_name], [qubits[i]])

        if transpiled:
            qc.barrier()
            qc.measure(qubits, qubits)

        self.instance = (qc, basis_operator, noise_repetitions, "".join(binstr))
        return self.instance

    def generate_measurement_circuits(self):
        num = len(self.pauli_strings)*len(self.depths)*self.samples
        n = self.n
        gates = self.transpiled_gates
        qubits = self.layer

        circs = []
        paulis = []
        lengths = []
        rostrings = []

        r=0
        for basis_operator,noise_repetitions,sample in product(self.pauli_strings, self.depths, range(self.samples)):
            qc = QuantumCircuit(self.num_qubits, self.num_qubits)

            #choose first layer
            twirl_layer = [Pauli(p) for p in choices("IXYZ",k=n)]
            #Choose B gates
            for i,(b,p) in enumerate(zip(basis_operator, twirl_layer)):
                gate_name = {'X':"H", 'Y':"HS"}.get(b,"")
                gate_name += p.to_label()
                qc = qc.compose(gates[gate_name], [qubits[i]])

            qc.barrier()
            #add the twirled layers
            for l in range(noise_repetitions-1):
                for i in range(n):
                    tw_op = choice("IXYZ")
                    qc = qc.compose(gates[tw_op], [qubits[i]])
                    twirl_layer[i] = twirl_layer[i].compose(Pauli(tw_op))
                qc.barrier()

            binstr = ''.join(choices('01',k=n))
    
            for i,(p,b,x) in enumerate(zip(twirl_layer, basis_operator, binstr)):
                gate_name = p.to_label()[-1]
                gate_name += {'X':'H', 'Y':"SdgH"}.get(b,"")
                gate_name += {'0':'', '1':'X'}[x]
                qc = qc.compose(gates[gate_name], [qubits[i]])

            qc.barrier()
            qc.measure(qubits, range(n))
            
            circs.append(qc)
            paulis.append(basis_operator)
            lengths.append(noise_repetitions)
            rostrings.append(binstr)
            r+= 1
            progressBar(r, num)

        self.metadata = [{"basis":p, "length":l, "rostring":binstr} for (p,l,binstr) in zip(paulis, lengths, rostrings)]
        return circs, self.metadata

class SingleGateLayerAnalysis:

    def __init__(self, layer, adjacency_matrix, samples, depths, backend):
        self.depths = depths
        self.samples = samples
        self.adjacency_matrix = adjacency_matrix
        self.backend = backend
        self.layer = layer

        self.n = len(layer)
        self.ordered_data = {}

        self.pauli_list = self.get_benchmark_paulis()
        self.fidelities = []
        self.model_coeffs = []
    
    @classmethod
    def fromExperiment(cls, experiment_class):
        return cls(
            layer = experiment_class.layer,
            adjacency_matrix = experiment_class.adjacency_matrix,
            samples = experiment_class.samples,
            depths = experiment_class.depths,
            backend = experiment_class.backend
        )

    def get_real_errors(self):
        errors = []
        noise_model = NoiseModel.from_backend(self.backend)._local_quantum_errors
        noise_model.pop("reset", None)
        noise_model.pop("x", None)
        noise_model.pop("id", None)
        for q in self.layer:
            r = 0
            op = SuperOp(np.zeros([4,4]))
            for instruction in noise_model.keys():
                error = noise_model[instruction]
                for qb in error.keys():
                    if len(qb) == 1 and q in qb:
                        r += 1
                        op += error[qb].to_quantumchannel()
            errors.append(op / r)

        return errors

    def get_real_PTM(self):
        errors = self.get_real_errors()
        n = self.n
        channel = SuperOp(np.identity(1))
        for error in errors:
            channel = channel.tensor(error)

        channel = channel.data
        twirled_channel = np.zeros([4**n,4**n])

        for s in pauli_basis(n, pauli_list=True):
            p = np.kron(np.conjugate(s.to_matrix()), s.to_matrix())
            twirled_channel = np.add(twirled_channel, 1/4**n * p @ channel @ p)

            transfer_matrix = PTM(SuperOp(twirled_channel)).data

        real_errors = {}
        for i,p in enumerate(pauli_basis(3, pauli_list=True).to_labels()):
            real_errors[p] = transfer_matrix[i][i]

        return real_errors

    def get_expectation(self, p, result, **data):
        counts = result
        ro_string = data["rostring"]
        expectation = 0
        shots = sum(counts.values())
        for count,freq in zip(counts.keys(), counts.values()):
            count = count[::-1][:self.n]
            count = [{('1','1'):'0', ('1','0'):'1'}.get((ro,key), key) for (ro,key) in zip(ro_string, count)]
            pauli_weight = [{'I':'0'}.get(char, '1') for char in p]
            sgn = sum([{('1','1'):1}.get((dig, p),0) for dig,p in zip(count,pauli_weight)])
            expectation += freq*(-1)**sgn
        return expectation/shots

    def get_benchmark_paulis(self):
        n = len(self.adjacency_matrix)
        pauli_list = []
        idPauli = ['I']*n    
    
        #get all single-weight paulis
        for i in range(n):
            for op in ['X','Y','Z']:
                pauli = idPauli.copy()
                pauli[i] = op
                pauli_list.append("".join(pauli))
                
        #get all weight-two paulis on nieghboring qubits
        for vert1,link in enumerate(self.adjacency_matrix):
            for vert2,val in enumerate(link[:vert1]):
                if val == 1:
                    for pauli1, pauli2 in product(['X','Y','Z'], repeat = 2):
                        pauli = idPauli.copy()
                        pauli[vert1] = pauli1
                        pauli[vert2] = pauli2
                        pauli_list.append("".join(pauli))

        return pauli_list

    def process_data(self, results, data):
        ordered_data = {}

        ordered_data = {p:{l:{"expectation":0, "total":0} for l in self.depths} for p in self.pauli_list}

        for p,(result, datum) in product(self.pauli_list, zip(results, data)):
            #check if non-identity terms in p overlap with instance
            if all([c1 == c2 for c1,c2 in zip(p, datum['basis']) if not c1 == 'I']):
                ordered_data[p][datum['length']]['expectation'] += self.get_expectation(p, result, **datum)/self.samples
                ordered_data[p][datum['length']]['total'] += 1/self.samples

        expfit = lambda x,a,b : a*np.exp(-x*b)
        p0 = [.85, .01]
        self.fidelities = []
        for p in self.pauli_list:
            expectations = []
            for l in self.depths:
                ordered_data[p][l]['expectation'] /= ordered_data[p][l]['total']
                expectations.append(ordered_data[p][l]['expectation'])
            
            popt, pcov = curve_fit(expfit, self.depths, expectations, p0=p0)
            ordered_data[p]["fit"] = (popt[0], popt[1])
            ordered_data[p]["fidelity"] = expfit(1, 1, popt[1])
            self.fidelities.append(ordered_data[p]["fidelity"])
        
        s_prod = lambda a,b : {True:0, False:1}[Pauli(a).commutes(Pauli(b))]
        M = [[s_prod(a,b) for a in self.pauli_list] for b in self.pauli_list]
        self.model_coeffs, rnorm = nnls(M, -.5*np.log(self.fidelities))
        
        for p,f in zip(self.pauli_list,self.fidelities):
            ordered_data[p]["coefficient"] = f

        self.ordered_data = ordered_data
        return ordered_data

    def plot_fits(self, subset = None):
        if subset == None:
            subset = self.pauli_list
        cycol = cycle("rgbcmk") 
        xrange = np.linspace(0, np.max(self.depths), 100)
        expfit = lambda x,a,b : a*np.exp(-x*b)
        for p in subset:
            c = next(cycol)
            plt.plot(self.depths, [self.ordered_data[p][l]['expectation'] for l in self.depths], c+'X')
            a,b = self.ordered_data[p]["fit"]
            plt.plot(xrange, [expfit(x, a, b) for x in xrange], c)
        plt.title("Pauli Fidelity vs. Noise Repetitions")
        plt.xlabel("Noise Repetitions")
        plt.ylabel("Fidelity")
        plt.show()
        plt.show()

    def plot_fidelities(self, subset = None):
        if subset == None:
            subset = self.pauli_list
        avg_transfer_matrix = self.get_real_PTM()
        measured = [100*(1-self.ordered_data[p]['fidelity']) for p in subset]
        real = [100*(1-avg_transfer_matrix[p]) for p in subset]
        ax = np.arange(len(subset))
        plt.bar(ax+.2, measured, .4, color='b')
        plt.bar(ax-.2, real, .4, color='r')
        plt.xticks(ax, subset)
        plt.title("Measured vs ideal fidelities")
        plt.xlabel("Pauli")
        plt.ylabel("1-100xf")
        plt.show()

    def plot_model_coeffs(self, subset = None):
        if subset == None:
            subset = self.pauli_list

        transfer_matrix = self.get_real_PTM()
        ideal_fids = [transfer_matrix[p] for p in self.pauli_list]
        s_prod = lambda a,b : {True:0, False:1}[Pauli(a).commutes(Pauli(b))]
        M = [[s_prod(a,b) for a in self.pauli_list] for b in self.pauli_list]
        coeffs, rnorm = nnls(M, -.5*np.log(ideal_fids))
        ax = np.arange(len(subset))
        plt.bar(ax+.2, self.model_coeffs, .4, color = 'm')
        plt.bar(ax-.2, coeffs, .4, color= 'g')
        plt.xticks(ax, subset) 
        plt.xlabel("Pauli")
        plt.ylabel("Model coefficient")
        plt.title("Measured vs ideal model coefficients")