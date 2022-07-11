import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from typing import List

'''
This class translates an adjacency list of random labels into an adjacency matrix,
and provides some utilities for getting information about the connectivity of the processor.
Uncommented version in processor_connectivity_tools.py
'''
class connectivity_map:

    def __init__(self, undirected_adjacency_list : List[tuple]):

        #For typographical reasons
        ual = undirected_adjacency_list
        #use set so that no two qubits are mapped twice
        qubits_unordered = set()
        #add all qubits from adjacency list to
        for (qubit1,qubit2) in ual:
            qubits_unordered.add(qubit1)
            qubits_unordered.add(qubit2)

        self.num_qubits = len(qubits_unordered)
        self.qubits = range(self.num_qubits) #for iterating
        #this dictionary stores the mapping from internal representation to hardware
        self.mapping = {i:qubit for (i,qubit) in enumerate(qubits_unordered)} #mapping from internal to hardware
        self.reverse_mapping = {v:i for (i,v) in self.mapping.items()} #mapping from hardware to internal
        self.mapped_adjacency_list = undirected_adjacency_list #hardware mapping
        #internal mapping
        self.unmapped_adjacency_list = [(self.reverse_mapping[qubit1], self.reverse_mapping[qubit2]) for (qubit1, qubit2) in ual]
        self.adjacency_matrix = [[] for i in self.qubits] #initialize adjacency matrix
        #change adjacency list to matrix
        for (qubit1,qubit2) in self.unmapped_adjacency_list:
            self.adjacency_matrix[qubit1].append(qubit2)
            self.adjacency_matrix[qubit2].append(qubit1)
    
    #a recursive internal method to find chains. Method has side effects but no return value
    def _get_chain(self, qubit : int, depth : int, prev_chain : List[int], chains : List[tuple]) -> None:
        if depth==0: #add chain as tuple and terminate
            chains.append(tuple(prev_chain)) 
            return

        #create new chain and add each child of current node
        for child in self.adjacency_matrix[qubit]:
            chain = prev_chain.copy() #copy old chain
            if not child in chain: #add child unless there is a loop
                #add child to chain and continue recursively
                chain.append(child)
                self._get_chain(child,depth-1, chain, chains)

    #this method returns all chains of adjacent qubits in the processor
    def get_chains(self, chain_length : int) -> List[tuple]:
        #although it could make sense to use a set, tuples are ordered and sets are unhashable
        chains = []
        #start recursion at every qubit
        for qubit in self.qubits:
            chain = [qubit]
            self._get_chain(qubit, chain_length-1, chain, chains)

        #remove any duplicate chains equal up to permutation
        i = 0
        N = len(chains) #will decrease as chains are removed
        while i < N:
            #log redundant chains for removal
            chains_to_remove = []

            #check all chains greater than i
            for j in range(i+1,N):
                if all(elem in chains[i] for elem in chains[j]): #check if chains are equal up to permutation
                    chains_to_remove.append(chains[j])
            #remove chains
            for j in chains_to_remove:
                chains.remove(j)
            #decrease N to reflect new length of set.
            #i does not need to be decreased becuause only chains > i are checked
            N -= len(chains_to_remove)
            i+=1

        return chains
    
    #This method returns all chains of length depth containing qubit
    def get_chains_with(self, qubit : int, depth : int) -> List[tuple]:
        #create head for recursion
        chains = []
        chain = [qubit]
        #start recursion with current qubit
        self._get_chain(qubit, depth, chain, chains)

        return chains

    #get all qubits that are separated by depth steps from the current qubit
    def connections_at_depth(self, qubit : int, depth : int) -> List[int]:
        #get the chains originating at qubit and return tail of each
        return [connection[-1] for connection in self.get_chains_with(qubit, depth)]

    #get all pairs of qubits that are separated by depth steps
    def all_connections_at_depth(self, depth : int) -> List[tuple]:
        #get all of the chains of length depth and return first and last elements of each
        return[(chain[0], chain[-1]) for chain in self.get_chains(depth+1)]

    #create a linear processor of a number of qubits
    def linear_proc(num_qubits : int):
        #create edges representing linear connectivity
        edges = [(i, i+1) for i in range(num_qubits-1)]
        return connectivity_map(edges) 

    #create square processor with side length
    def square_proc(side_length : int):
        #linear processor, but connect ends for a cycle
        edges = [(i, i+1) for i in range((side_length-1)*4-1)]+[((side_length-1)*4-1, 0)]
        return connectivity_map(edges)
    
    #use networkx to display processor as graph
    def draw_processor(self) -> None:
        G = nx.Graph()
        G.add_edges_from(self.mapped_adjacency_list)
        nx.draw_networkx(G)
        plt.show()

    #Visualize a full-rank pauli operator applied to the processor
    #useful for verifying the model generation
    def draw_pauli(self, str : str) -> None:
        G = nx.Graph()
        G.add_edges_from(self.mapped_adjacency_list)
        pos = nx.spring_layout(G, seed=3113794652)
        #create alias labels corresponding to paulis
        labels = dict(zip(self.qubits, str))
        nx.draw_networkx(G, labels = labels, node_shape = "d", node_color = "w")
        plt.show()
    
    #return all paulis with support on qubit_group
    def benchmark_paulis(self, qubit_group : tuple) -> List[str]:
        weight = len(qubit_group)
        pauli_list = []
        #iterate through all weight-length combinations of nonidentity paulis
        for str in product(['X','Y','Z'], repeat = weight):
            pauli_str = ['I']*self.num_qubits #initialize all slots to I
            for char,qb in zip(str,qubit_group):
                pauli_str[qb] = char #apply pauli char to qubit qb
            pauli_list.append("".join(pauli_str)) #add pauli string to list

        return pauli_list #return list of pauli strings

    #run benchmark paulis on all chains of a certain weight
    def proc_benchmark_paulis_weight(self, weight : int) -> set:
        return set([pauli for group in self.get_chains(weight) for pauli in self.benchmark_paulis(group)])

    #run proc_benchmark_paulis_weight for all chains up to max_weight
    def proc_benchmark_max_weight(self, max_weight) -> List[str]:
        benchmark_paulis = set()
        for weight in range(1,max_weight+1):
            benchmark_paulis = benchmark_paulis.union(self.proc_benchmark_paulis_weight(weight))
        return list(benchmark_paulis)