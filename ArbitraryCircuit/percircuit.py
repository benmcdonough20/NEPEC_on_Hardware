from circuitlayer import CircuitLayer
import contextvars

imp = contextvars.ContextVar("implementation")

class PERCircuit:
    def __init__(self, qc):
        self.n = qc.num_qubits()
        self._qc = qc
        self._layers = self._circuit_to_benchmark_layers()
        self._measurements = self._get_measurements()
        self.reps = set(self._layers)

    def _get_measurements(self):
        qc = self._qc
        measurements = qc.copy_empty()
        for inst in qc:
            if inst.name() == "measure":
                measurements.append_instruction(inst)
        return measurements

    def _circuit_to_benchmark_layers(self):
        layers = []
        qc = self._qc
        inst_list = [inst for inst in qc if inst.name() != "measure"] 
        while inst_list:
            circ = qc.copy_empty()
            layer_qubits = set()
            for inst in inst_list.copy():
                if not layer_qubits.intersection(inst.support()):
                    if inst.weight() == 2:
                        layer_qubits = layer_qubits.union(inst.support())
                    circ.add_instruction(inst)
                    inst_list.remove(inst)
            layers.append(CircuitLayer(circ))

        return layers
    
    def sample_PER(self, noise_strength):
        """ This method uses the learned noise model to sample
        PER representations of the circuit"""

        pass 