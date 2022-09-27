from circuitlayer import CircuitLayer

class PERCircuit:
    """Aggregation of circuit layers. Responsable for generating circuit layers and
    representing them. The noise model is provided to this object to generate PER samples
    circuits"""

    def __init__(self, qc):
        self.n = qc.num_qubits()
        self._qc = qc
        self._layers = self._circuit_to_benchmark_layers()

    def add_noise_models(self, noise_data_frame):
        for layer in self._layers:
                layer.noisemodel = noise_data_frame.noisemodels[layer.cliff_layer]


    def _circuit_to_benchmark_layers(self):
        layers = []
        qc = self._qc
        inst_list = [inst for inst in qc if not inst.ismeas()] 
        while inst_list:
            circ = qc.copy_empty()
            layer_qubits = set()
            for inst in inst_list.copy():
                if not layer_qubits.intersection(inst.support()):
                    circ.add_instruction(inst)
                    inst_list.remove(inst)

                if inst.weight() == 2:
                    layer_qubits = layer_qubits.union(inst.support())

            newlayer = CircuitLayer(circ)
            if newlayer.cliff_layer:
                layers.append(CircuitLayer(circ))

        return layers
    
    def display_circuit(self):
        circ = self._qc.copy_empty()
        for layer in self._layers:
            circ.compose(layer.single_layer)
            circ.compose(layer.cliff_layer)
        return circ.original()

    def __getitem__(self, item):
        return self._layers.__getitem__(item)