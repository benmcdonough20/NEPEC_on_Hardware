from circuitlayer import CircuitLayer

class PERCircuit:
    """Aggregation of circuit layers. Responsable for generating circuit layers and
    representing them. The noise model is provided to this object to generate PER samples
    circuits"""

    def __init__(self, qc):
        self.n = qc.num_qubits()
        self._qc = qc
        self._layers = self._circuit_to_benchmark_layers()
        self._measurements = self._get_measurements()

    def _get_measurements(self):
        qc = self._qc
        measurements = qc.copy_empty()
        for inst in qc:
            if inst.name() == "measure":
                measurements.append_instruction(inst)
        return measurements

    def add_noisemodel(self, noise_model):
        for layer in self._layers:
            if layer.cliff_layer == noise_model.cliff_layer:
                layer.noisemodel = noise_model

    def _circuit_to_benchmark_layers(self):
        layers = []
        qc = self._qc
        inst_list = [inst for inst in qc if inst.name() != "measure"] 
        while inst_list:
            circ = qc.copy_empty()
            layer_qubits = set()
            for inst in inst_list.copy():
                if not layer_qubits.intersection(inst.support()):
                    circ.add_instruction(inst)
                    inst_list.remove(inst)

                if inst.weight() == 2:
                    layer_qubits = layer_qubits.union(inst.support())
            layers.append(CircuitLayer(circ))

        return layers
    
    def sample_PER(self, noise_strength):
        """ This method uses the learned noise model to sample
        PER representations of the circuit"""

        pass 

    def display_circuit(self):
        circ = self._qc.copy_empty()
        for layer in self._layers:
            circ.compose(layer.single_layer)
            circ.compose(layer.cliff_layer)
        return circ