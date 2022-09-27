"""Take in a per circuit with expectation value and return a single sampled instance"""
from instance import Instance

class PERInstance(Instance):

    def __init__(
        self, 
        processor,
        inst_map,
        percirc, 
        meas_basis, 
        noise_strength):
        
        self._percirc = percirc
        self._processor = processor
        self.noise_strength = noise_strength
        self.meas_basis = meas_basis
        self.pauli_type = percirc._qc.pauli_type
        self._inst_map = inst_map

        super().__init__(percirc._qc, meas_basis)

    def _instance(self): #called automatically by super().__init__
        
        self._circ = self._circ.copy_empty()
        circ = self._circ
        n = circ.num_qubits()
        p_type = self.pauli_type
        self.sgn_tot = 0
        self.overhead = 1
        
        for layer in self._percirc:
            sgn = layer.sample(self.noise_strength,circ)
            self.sgn_tot ^= sgn
            self.overhead *= layer.noisemodel.overhead 

        self._basis_change()
        self._readout_twirl()
        #self._rostring = self.pauli_type.random(n, subset = "IX").to_label()
        
        circ.measure_all()
        #self._circ = self._processor.transpile(self._circ, self._inst_map)

    def get_adjusted_expectation(self, pauli):
        expec = self.get_expectation(pauli)
        return expec * self.overhead * (-1)**self.sgn_tot