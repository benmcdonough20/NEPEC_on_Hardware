from primitives.circuit import QiskitCircuit
from percircuit import PERCircuit
from perrun import PERRun
from primitives.processor import QiskitProcessor

class PERExperiment:
    
    def __init__(self, circuits, inst_map, noise_data_frame, backend):
        circuit_interface = None

        if circuits[0].__class__.__name__ == "QuantumCircuit":
            circuit_interface = QiskitCircuit
            self._processor = QiskitProcessor(backend)
        else:
            raise Exception("Unsupported circuit type")

        self.noise_data_frame = noise_data_frame
        self.spam = noise_data_frame.spam

        per_circuits = []
        self.pauli_type = circuit_interface(circuits[0]).pauli_type
        for circ in circuits:
            circ_wrap = circuit_interface(circ)
            per_circ = PERCircuit(circ_wrap)
            per_circ.add_noise_models(noise_data_frame)
            per_circuits.append(per_circ)

        self._per_circuits = per_circuits
        self._inst_map = inst_map


    def get_meas_bases(self, expectations):

        meas_bases = []
        for pauli in expectations:
            for i,base in enumerate(meas_bases):
                if base.nonoverlapping(pauli):
                    meas_bases[i] = base.get_composite(pauli)
                    break
            else:
                meas_bases.append(pauli)

        self.meas_bases = meas_bases


    def generate(self, noise_strengths, expectations, samples):

        expectations = [self.pauli_type(label) for label in expectations] 

        self.get_meas_bases(expectations)
        bases = self.meas_bases

        self._per_runs = []
        for pcirc in self._per_circuits:
            per_run = PERRun(self._processor, self._inst_map, pcirc, samples, noise_strengths, bases, expectations, self.spam)
            self._per_runs.append(per_run)

    def run(self, executor):
        instances = []
        for run in self._per_runs:
            instances += run.instances
        
        circuits = [inst.get_circuit() for inst in instances] 
        results = executor(circuits)
        
        for inst, res in zip(instances, results):
            inst.add_result(res)

    def analyze(self):
        for run in self._per_runs:
            run.analyze()