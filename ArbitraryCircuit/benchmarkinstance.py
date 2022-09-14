import contextvars
from random import choices
from primitives.term import QiskitPauli

pauli_type = QiskitPauli

class BenchmarkInstance:
    def __init__(self, prep_basis, meas_basis, noise_repetitions, procspec, cliff_layer):
        self.prep_basis = prep_basis
        self.meas_basis = meas_basis
        self.cliff_layer = cliff_layer
        self.noise_repetitions = noise_repetitions
        self.type = None

        self.circuit = self.instance(procspec)

    def instance(self, procspec):

            circ = self.cliff_layer.copy_empty() #storing the final circuit
            n = self.cliff_layer.num_qubits()
            pauli_frame = pauli_type.ID(n) 

            #apply the prep operators to the circuit
            circ.compose(self.prep_basis.z_basis(circ))

            #apply repetitions of noise, including basis-change gates when needed
            for i in range(self.noise_repetitions):

                twirl = pauli_type.random(n)
                pauli_frame *= twirl 
                pauli_frame = pauli_frame.conjugate(self.cliff_layer)

                circ.add_pauli(twirl)

                circ.compose(self.cliff_layer)
                circ.barrier()

            #choose sstring of bit flips for readout twirling
            self.rostring = "".join(choices(['I','X'], k=n))
            circ.add_pauli(pauli_frame)

            circ.compose(self.meas_basis.z_basis(circ))
            circ.add_pauli(pauli_type(self.rostring))

            circ.measure(circ.qubits())

            return circ 

    def get_expectation(self, pauli):
        estimator = 0
        counts = self.result.counts
        rostring = self.rostring
        #compute locations of non-idetity terms (reversed indexing)
        pz = list(reversed([{pauli_type("I"):'0'}.get(p,'1') for p in pauli]))
        #compute estimator
        for key in counts.keys():
            #untwirl the readout
            ro_untwirled = [{'0':'1','1':'0'}[bit] if flip=="X" else bit for bit,flip in zip(key,rostring)]
            #compute the overlap in the computational basis
            sgn = sum([{('1','1'):1}.get((pauli_bit, key_bit), 0) for pauli_bit, key_bit in zip(pz, ro_untwirled)])
            #update estimator
            estimator += (-1)**sgn*counts[key]

        return estimator/sum(counts.values())