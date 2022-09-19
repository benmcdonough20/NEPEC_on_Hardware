from abc import ABC, abstractmethod
from primitives.result import Result
from qiskit.providers.backend import BackendV2
from qiskit import transpile
from primitives.circuit import Circuit, QiskitCircuit

class Processor(ABC):
    @abstractmethod
    def sub_map(self, qubits):
        pass

    @abstractmethod
    def transpile(self, circuit : Circuit, **kwargs):
        pass
    
    @abstractmethod
    def run(self, *circuits : Circuit, **kwargs) -> Result:
        pass

class QiskitProcessor(Processor):
    
    def __init__(self, backend : BackendV2):
        self._qpu = backend

    def sub_map(self, inst_map):
        return self._qpu.coupling_map.graph.subgraph(inst_map)

    def transpile(self, circuit : QiskitCircuit, inst_map = None, **kwargs):
        return QiskitCircuit(transpile(circuit.qc, self._qpu, inst_map = inst_map, **kwargs))

    def run(self, circuit : QiskitCircuit, **kwargs) -> Result:
        counts = self._qpu.run(circuit, **kwargs).result().get_counts()
        return Result(counts)