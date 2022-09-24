from random import choices
from primitives.pauli import QiskitPauli
from processorspec import ProcessorSpec
from primitives.circuit import Circuit

import logging
logger = logging.getLogger("experiment")

SINGLE = 1
PAIR = 2

class BenchmarkInstance:
    """Generates a benchmark instance implementing basis change gates, noise layer repetitions,
    Pauli twirling, and readout twirling. Stores circuit and corresponding metadata, including
    the layer to which it pertains. Fundamentally the experiment consists only of a list of
    benchmarkinstances."""

    def __init__(
        self, 
        prep_basis : QiskitPauli, 
        meas_basis : QiskitPauli, 
        noise_repetitions : int, 
        procspec : ProcessorSpec, 
        cliff_layer : Circuit, 
        type = PAIR
        ):

        self.prep_basis = prep_basis #preparation bases
        self.meas_basis = meas_basis #measurement basis
        self.cliff_layer = cliff_layer #Clifford layer profile associated with noise
        self.depth = noise_repetitions #number of repetitions of noisy layer
        self.type = type #Whether circuit is instance of pair or single measurement

        self._circuit = self._instance(procspec)

    def _instance(self, procspec):
            #Generates a circuit for benchmarking. Takes as input the processor specification
            #in case piecewise transpilation is necessary.

            circ = self.cliff_layer.copy_empty() #storing the final circuit
            n = self.cliff_layer.num_qubits()
            pauli_type = circ.pauli_type
            pauli_frame = pauli_type.ID(n) 

            #apply the prep operators to the circuit
            circ.compose(self.prep_basis.basis_change(circ))

            #apply repetitions of noise, including basis-change gates when needed
            for i in range(self.depth):

                twirl = pauli_type.random(n)
                pauli_frame *= twirl 
                pauli_frame = self.cliff_layer.conjugate(pauli_frame)

                circ.add_pauli(twirl)

                circ.compose(self.cliff_layer)
                circ.barrier()

            #choose string of bit flips for readout twirling
            self.rostring = "".join(choices(['I','X'], k=n))
            circ.add_pauli(pauli_frame)

            #Add basis change and readout twirling
            circ.compose(self.meas_basis.basis_change(circ).inverse())
            circ.add_pauli(pauli_type(self.rostring))

            #add measurements to the circuit and transpile
            circ.measure_all()
            circ = procspec.transpile(circ)

            return circ 
    
    def add_result(self, result):
        """Takes a counter object with binary strings as keys and frquencies as values. Untwirls
        the readout."""
        self.ro_untwirled = {}
        rostring = self.rostring
        for key in result:
            newkey = "".join([{'0':'1','1':'0'}[bit] if flip=="X" else bit for bit,flip in zip(key,rostring)])
            self.ro_untwirled[newkey] = result[key]

        self.shots = sum(self.ro_untwirled.values())

    def get_expectation(self, pauli):
        """Return the expectation of a pauli operator after a measurement of the circuit,
        adjusting the result for the readout twirling"""

        pauli_type = self.cliff_layer.pauli_type
        estimator = 0
        ro_untwirled = self.ro_untwirled
        #compute locations of non-idetity terms (reversed indexing)
        pz = list(reversed([{pauli_type("I"):'0'}.get(p,'1') for p in pauli]))
        #compute estimator
        for key in ro_untwirled.keys():
            #compute the overlap in the computational basis
            sgn = sum([{('1','1'):1}.get((pauli_bit, key_bit), 0) for pauli_bit, key_bit in zip(pz, key)])
            #update estimator
            estimator += (-1)**sgn*ro_untwirled[key]

        return estimator/self.shots