from processorspec import ProcessorSpec
from layerlearning import LayerLearning
from analysis import Analysis
from percircuit import PERCircuit
import logging
 
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from primitives.circuit import QiskitCircuit
from primitives.processor import QiskitProcessor
import pickle

class SparsePauliTomographyExperiment:
    """This class carries out the full experiment by creating and running a LayerLearning
    instance for each distinct layer, running the analysis, and then returning a PERCircuit
    with NoiseModels attached to each distinct layer"""

    def __init__(self, circuit, inst_map, backend):

        if circuit.__class__.__name__ == "QuantumCircuit":
            circuit_interface = QiskitCircuit(circuit)
            backend_interface = QiskitProcessor(backend)  
        else:
            raise Exception("Unsupported circuit type")
        
        self._circuit = PERCircuit(circuit_interface)
        self._profiles = set(self._circuit._layers)

        logger.info("Generated layer profile with %s layers"%len(self._profiles))

        self._procspec = ProcessorSpec(inst_map, backend_interface)
        self.instances = []

    def generate(self, samples, single_samples, depths):

        self.instances = []
        for l in self._profiles:
            learning = LayerLearning(l, samples, single_samples, depths)
            self.instances += learning.procedure(self._procspec)

        self.analysis = Analysis(self.instances, self._procspec)

    def run(self, executor):
        for l in self._profiles:
            circuits = [inst._circuit.original() for inst in self.instances]

        results = executor(circuits)

        for res,inst in zip(results, self.instances):
            #TODO: feed into results object
            inst.result = res

    def get_profiles(self):
        return [circ.original() for circ in self._profiles]

    def analyze(self):
        """Runs analysis on each layer representative and stores for later plotting/viewing"""
        self.analysis.analyze()

    def save(self):
        """Save experiment to be processed later"""
        with open("experiment.dat", "wb") as f:
            pickle.dump(self, f)
   
    @staticmethod 
    def load():
        with open("experiment.dat", "rb") as f:
            return pickle.load(f)