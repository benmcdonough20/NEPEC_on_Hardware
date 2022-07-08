import networkx as nx
import matplotlib.pyplot as plt
from itertools import product

class connectivity_map:
    def __init__(self, undirected_adjacency_list):
        self.undirected_adjacency_list = undirected_adjacency_list
        ual = undirected_adjacency_list
        qubits_unordered = set()
        for (qubit1,qubit2) in ual:
            qubits_unordered.add(qubit1)
            qubits_unordered.add(qubit2)

        self.mapping = {i:qubit for (i,qubit) in enumerate(qubits_unordered)}
        self.reverse_mapping = {v:i for (i,v) in self.mapping.items()}
        self.num_qubits = len(qubits_unordered)
        self.qubits = range(self.num_qubits)
        self.connections = [[] for i in range(self.num_qubits)]
        ual = [(self.reverse_mapping[qubit1], self.reverse_mapping[qubit2]) for (qubit1, qubit2) in ual]
        for i,(qubit1,qubit2) in enumerate(ual):
            self.connections[qubit1].append(qubit2)
            self.connections[qubit2].append(qubit1)
        
    def get_chains(self, chain_length):
        chains = []
        for qubit in self.qubits:
            chain = [qubit]
            self._get_chain(qubit, chain_length-1, chain, chains)
        i = 0
        N = len(chains)
        while i < N:
            chains_to_remove = []
            for j in range(i+1,N):
                if all(elem in chains[i] for elem in chains[j]):
                    chains_to_remove.append(chains[j])
            for j in chains_to_remove:
                chains.remove(j)
            N -= len(chains_to_remove)
            i+=1
        return chains
    
    def connections_with_depth(self, qubit, depth):
        chains = []
        chain = [qubit]
        self._get_chain(qubit, depth, chain, chains)
        return chains

    def _get_chain(self, qubit,depth, prev_chain, chains):
        if depth==0:
            chains.append(tuple(prev_chain))
            return
        for child in self.connections[qubit]:
            chain = prev_chain.copy()
            if not child in chain:
                chain.append(child)
                self._get_chain(child,depth-1, chain, chains)
    
    def connections_at_depth(self, qubit, depth):
        return [connection[-1] for connection in self.connections_with_depth(qubit, depth)]

    def all_connections_at_depth(self, depth):
        return[(chain[0], chain[-1]) for chain in self.get_chains(depth+1)]

    def linear_proc(num_qubits):
        edges = [(i, i+1) for i in range(num_qubits-1)]
        return connectivity_map(edges) 

    def square_proc(side_length):
        edges = [(i, i+1) for i in range((side_length-1)*4-1)]+[((side_length-1)*4-1, 0)]
        return connectivity_map(edges)
    
    def draw_processor(self):
        G = nx.Graph()
        G.add_edges_from(self.undirected_adjacency_list)
        nx.draw_networkx(G)
        plt.show()

    def draw_pauli(self, str):
        G = nx.Graph()
        G.add_edges_from(self.undirected_adjacency_list)
        pos = nx.spring_layout(G, seed=3113794652)
        labels = dict(zip([self.mapping[qubit] for qubit in self.qubits],str))
        nx.draw_networkx(G, labels = labels, node_shape = "d", node_color = "w")
        plt.show()
        
    def benchmark_paulis(self, qubit_group):
        weight = len(qubit_group)
        pauli_list = []
        for str in product(['X','Y','Z'], repeat = weight):
            pauli_str = ['I']*self.num_qubits
            for char,qb in zip(str,qubit_group):
                pauli_str[qb] = char
            pauli_list.append("".join(pauli_str))

        return pauli_list

    def proc_benchmark_paulis_weight(self, weight):
        return set([pauli for group in self.get_chains(weight) for pauli in self.benchmark_paulis(group)])

    def proc_benchmark_max_weight(self, max_weight):
        benchmark_paulis = set()
        for weight in range(1,max_weight+1):
            benchmark_paulis = benchmark_paulis.union(self.proc_benchmark_paulis_weight(weight))
        return list(benchmark_paulis)