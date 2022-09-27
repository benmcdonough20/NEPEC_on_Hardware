from random import choices

class Instance:
    """This class implements the framework that both the benchmark instances and per instances
    extend. This involves storing a circuit, changing to a pauli basis, measuring, twirling the readout,
    storing the result data, untwirling the readout, and computing expectation values."""

    def __init__(self, qc, meas_basis):
        self._circ = qc
        self._result = None
        self._rostring = None
        self._meas_basis = meas_basis

        self._instance()

    def _instance(self): 
        self._basis_change()
        self._readout_twirl()
        self._circ.measure_all()

    def add_result(self, result):
        if self._result:
            self._result = {**self._result, **result}
        else:
            self._result = result

    def get_circuit(self):
        """Returns a copy of the circuit to be run on the hardware in the native format"""
        return self._circ.original()

    def _readout_twirl(self):
        pauli_type = self._circ.pauli_type
        readout_twirl = pauli_type.random(self._circ.num_qubits(), subset = "IX")
        self._rostring = readout_twirl.to_label()
        self._circ.add_pauli(readout_twirl)

    def _basis_change(self):
        self._circ.compose(self.meas_basis.basis_change(self._circ).inverse())
        
    def _untwirl_result(self):
        ro_untwirled = {}
        rostring = self._rostring
        for key in self._result:
            newkey = "".join([{'0':'1','1':'0'}[bit] if flip=="X" else bit for bit,flip in zip(key,rostring)])
            ro_untwirled[newkey] = self._result[key]

        return ro_untwirled 

    def get_expectation(self, pauli):
        """Return the expectation of a pauli operator after a measurement of the circuit,
        adjusting the result for the readout twirling"""

        pauli_type = self._circ.pauli_type
        estimator = 0
        result = self._untwirl_result()
        #compute locations of non-idetity terms (reversed indexing)
        pz = list(reversed([{pauli_type("I"):'0'}.get(p,'1') for p in pauli]))
        #compute estimator
        for key in result.keys():
            #compute the overlap in the computational basis
            sgn = sum([{('1','1'):1}.get((pauli_bit, key_bit), 0) for pauli_bit, key_bit in zip(pz, key)])
            #update estimator
            estimator += (-1)**sgn*result[key]

        return estimator/sum(result.values())