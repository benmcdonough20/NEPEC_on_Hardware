from qiskit_experiments.framework import BaseExperiment, BaseAnalysis, ExperimentData, AnalysisResultData
from qiskit.quantum_info import Pauli
from qiskit.providers import Backend, Job
from typing import Sequence, List, Tuple, Dict
from qiskit import QuantumCircuit, execute
import matplotlib
import numpy as np
from numpy.random import rand
from qiskit.providers.options import Options
from qiskit.providers.aer.noise import NoiseModel

class PauliExperiment(BaseExperiment):
    
    signs = []
    model_coefficients = []
    model_paulis = []
    
    def __init__(
        self, 
        model_coefficients : Sequence[float],
        model_paulis : Sequence[Pauli],
        gate_to_mitigate : Sequence[Pauli], 
        num_samples : int,
        backend : Backend,
        ):

        self.model_coefficients = model_coefficients
        self.model_paulis = model_paulis
        self.gate_to_mitigate = gate_to_mitigate
        self.num_samples = num_samples

        self.signs = [None] * num_samples #sign metadata for analysis

        analysis = PauliAnalysis()

        super().__init__(
            qubits = [0],
            analysis = analysis,
            backend = backend,
            experiment_type = "Pauli Experiment"
        )

    def circuits(self) -> List[QuantumCircuit]:
        circuits = [None] * self.num_samples
        num = len(self.model_paulis)
        #Follow the procedure for sampling gates
        for i in range(self.num_samples):
            sgn = 1 #m keeps track of the sign, with paulis carrying a negative sign
            op = self.gate_to_mitigate 
            for j in range(num):
                if rand() < 1-self.model_coefficients[j]:
                    #with probability 1-\omega_k, sample the Pauli gate and compose into operator
                    op = op.compose(self.model_paulis[j]) #Wow I spent so long on this silly line
                    sgn *= -1
            qc = QuantumCircuit(1,1)
            qc.append(op.to_instruction(),[0])
            qc.measure(0,0)
            circuits[i] = qc
            self.signs[i] = sgn

        return circuits
    
    def _metadata(self) -> Dict[str, any]:
        overhead = np.product(np.subtract(np.multiply(2, self.model_coefficients), 1))
        return {"signs": self.signs, "overhead" : overhead}
    
    def _default_run_options(csl) -> Options:
        return Options(shots = 1, optimization_level = 0)
    
    def _run_jobs(self, circuits: List[QuantumCircuit], **run_options) -> List[Job]:
        """Run circuits on backend as 1 or more jobs."""
        # Run experiment jobs
        max_experiments = getattr(self.backend.configuration(), "max_experiments", None)
        if max_experiments and len(circuits) > max_experiments:
            # Split jobs for backends that have a maximum job size
            job_circuits = [
                circuits[i : i + max_experiments] for i in range(0, len(circuits), max_experiments)
            ]
        else:
            # Run as single job
            job_circuits = [circuits]

        # Run jobs
        jobs = [execute(circs, self.backend, **run_options) for circs in job_circuits]

        return jobs
        

class PauliAnalysis(BaseAnalysis):
    def __init__(self):
        super().__init__()
    
    def _run_analysis(self, experiment_data: ExperimentData) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        
        signs = experiment_data.metadata["signs"]
        overhead = experiment_data.metadata["overhead"]
        counts = experiment_data.jobs()[0].result().get_counts()
        results = [count.get('0',0)-count.get('1', 0) for count in counts]

        total = np.dot(results, signs)/len(results)
        #total = np.sum(results)/len(results)

        return ([AnalysisResultData("total", total)],[])
