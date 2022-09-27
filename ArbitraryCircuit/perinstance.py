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

        for layer in percirc:
            layer.noisemodel.init_scaling(noise_strength)
        
        super().__init__(percirc._qc, meas_basis)

    def _instance(self): #called automatically by super().__init__
        
        self._circ = self._circ.copy_empty()
        circ = self._circ
        n = circ.num_qubits()
        p_type = self.pauli_type
        self.sgn_tot = 0
        pauli_frame = p_type.ID(n)
        self.overhead = 1
        for layer in self._percirc:
            layer.noisemodel.init_scaling(self.noise_strength) 
            self.overhead *= layer.noisemodel.overhead 
            circ.compose(layer.single_layer)
            twirl = p_type.random(n)
            circ.add_pauli(twirl)
            op, sgn = layer.noisemodel.sample()
            self.sgn_tot ^= sgn
            circ.add_pauli(op)
            twirl = layer.cliff_layer.conjugate(twirl)
            pauli_frame *= twirl
            circ.compose(layer.cliff_layer)
            circ.barrier()


        circ.add_pauli(pauli_frame)
        self._basis_change()
        self._readout_twirl()
        
        circ.measure_all()
        #self._circ = self._processor.transpile(self._circ, self._inst_map)

    def get_adjusted_expectation(self, pauli):
        expec = self.get_expectation(pauli)
        return expec * self.overhead * (-1)**self.sgn_tot