from processorspec import ProcessorSpec
from layerlearning import LayerLearning
from percircuit import PERCircuit
from analysis import LayerAnalysis

class SparsePauliTomographyExperiment:
    """This class carries out the full experiment by creating and running a LayerLearning
    instance for each distinct layer, running the analysis, and then returning a PERCircuit
    with NoiseModels attached to each distinct layer"""

    def __init__(self, circuit, inst_map, backend):
        self._circuit = PERCircuit(circuit)
        self._layers = {}
        self._procspec = ProcessorSpec(inst_map, backend)
        self.instances = []
        
        for l in self._circuit.reps:
            self._layers[l] = LayerLearning(l, inst_map, self._procspec)

    def generate(self):
        self.instances = []
        for l in self._circuit.reps:
            self.instances += self._layers[l].procedure(self._procspec)

    def run(self, executor):
        for l in self._circuit.reps:
            circuits = [inst.circuit().original() for inst in self.instances]
        results = executor(circuits)
        for res,circ in zip(results, self.instances):
            circ.result = res

    def analyze(self):
        analyses = []
        for l in self._circuit.reps:
            instances = [inst for inst in self.instances if inst.cliff_layer == l]
            analyses.append(LayerAnalysis(l, instances, self._procspec))


    def save(self):
        """Save experiment to be processed later"""

        pass