from noisemodel import NoiseModel
from benchmarkinstance import BenchmarkInstance, PAIR, SINGLE
from itertools import product
import logging

logger = logging.getLogger("experiment")


class LayerLearning:
    def __init__(self, cliff_layer, samples, single_samples, depths):
        self._cliff_layer = cliff_layer
        self.samples = samples
        self.depths = depths
        self.single_samples = single_samples

    def procedure(self, procspec):
        
        def issingle(term):
            pair = self._cliff_layer.conjugate(term)
            return pair == term or pair not in procspec.model_terms

        #find disjoint operators that can be measured simultaneously to find six bases
        pairs = set([frozenset([p,self._cliff_layer.conjugate(p)]) for p in procspec.model_terms if not issingle(p)])
        single_bases = []
        for p1,p2 in pairs:
            for i,pauli in enumerate(single_bases):
                pair = self._cliff_layer.conjugate(pauli)
                if pauli.separate(p1) and pair.separate(p2):
                    single_bases[i] = pauli * p2
                    break
            else:
                single_bases.append(p2)

        logger.info("Chose single bases:")
        logger.info([str(p) for p in single_bases])

        instances = []

        for basis,d,s in product(procspec.meas_bases, self.depths, range(self.samples)):
            inst = BenchmarkInstance(basis, basis, d, procspec, self._cliff_layer)
            instances.append(inst)

        for basis,s in product(single_bases, range(self.single_samples)):
            inst = BenchmarkInstance(self._cliff_layer.conjugate(basis), basis, SINGLE, procspec, self._cliff_layer, SINGLE)
            instances.append(inst)

        logger.info("Created experiment consisting of %u instances"%len(instances))
        self.instances = instances 
        return instances