from abc import abstractmethod, ABC
from itertools import product

class Pauli(ABC):

    @abstractmethod
    def __init__(self, label):
        pass

    @abstractmethod
    def conjugate(self, clifford):
        pass

    @abstractmethod
    def weight(self):
        pass

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def z_basis(self, qc):
        pass

    @abstractmethod
    def to_label(self):
        pass

    def disjoint(self, other):
        return all([p1==p2 or (p1 == self.ID or p2 == self.ID) for p1,p2 in zip(self, other)])

    def self_conjugate(self, clifford):
        return self == self.conjugate(clifford)

    @classmethod
    def random(cls, n, subset = "IXYZ"):
        return cls("".join(choices(subset, k=n)))

    @classmethod
    def ID(cls, n):
        return cls("I"*n)
    
    def __str__(self):
        return self.to_label()

from qiskit.quantum_info import Pauli as qiskit_pauli
from random import choices

class QiskitPauli(Pauli):

    def __init__(self, name):
        self._pauli = qiskit_pauli(name)

    def conjugate(self, clifford):
        return QiskitPauli(self._pauli.evolve(clifford.qc).to_label())
    
    def weight(self):
        return len(self._pauli)

    def __mul__(self, other):
        return QiskitPauli(self._pauli.compose(other._pauli).to_label())

    def __eq__(self, other):
        return self._pauli.x == other._pauli.x and self._pauli.z == other._pauli.z

    def __getitem__(self, item):
        return QiskitPauli(self._pauli.__getitem__(item))

    def to_label(self):
        return self._pauli.to_label()

    def __str__(self):
        return self.to_label()

    def z_basis(self, qc):
        circ = qc.copy_empty()
        for p,q in zip(self,qc.qc.qubits):
            if p == QiskitPauli("X"):
                circ.qc.h(q)
            elif p == QiskitPauli("Y"):
                circ.qc.h(q)
                circ.qc.s(q)

        return circ