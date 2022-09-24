from abc import abstractmethod, ABC
from random import choices

class Pauli(ABC):
    """An abstract class used as a wrapper for the native implementation of Pauli algebra.
    Whatever Pauli representation is chosen needs to be able to move back and forth between
    the Pauli string representation, and interact with circuit objects."""

    @abstractmethod
    def __init__(self, label : str) -> None:
        """Initialize a Pauli using a string consisting of the characters IXYZ"""

    @abstractmethod
    def weight(self) -> int:
        """Return the number of qubits nontrivially affected by the Pauli operator"""

    @abstractmethod
    def basis_change(self, qc) -> None:
        """This method appends to the circuit qc instructions to change from the computational
        basis to the eigenbasis of the Pauli operator"""

    @abstractmethod
    def to_label(self) -> str:
        """Return a label representation of the Pauli operator without phase"""

    @abstractmethod
    def commutes(self, other) -> bool:
        """Returns true if Pauli commutes with other"""

    @abstractmethod
    def __mul__(self, other):
        """Implement multiplication of one Pauli by another Pauli, ignoring phase"""

    @abstractmethod
    def __getitem__(self, item):
        """Used to iterate through the terms in the operator"""

    def simultaneous(self, other): #returns True if other can be measured simultaneously with self
        return all([p1==p2 or p2 == self.ID(1) for p1,p2 in zip(self, other)])

    def separate(self, other): #returns True if two Paulis have disjoin supports
        return all([p1==p2 or p2 == self.ID(1) or p1 == self.ID(1) for p1,p2 in zip(self, other)])

    def self_conjugate(self, clifford): #returns True if self is an eigenoperator of clifford
        return self == clifford.conjugate(self)

    @classmethod
    def random(cls, n, subset = "IXYZ"): #generates a random Pauli operator
        return cls("".join(choices(subset, k=n)))

    @classmethod
    def ID(cls, n): #returns an identity operator of the desired rank
        return cls("I"*n)
    
    def __hash__(self): #hashing does not record phase if any
        return self.to_label().__hash__()

    def __eq__(self, other): #equality does not record phase if any
        return self.to_label() == other.to_label()

    def __repr__(self):
        return self.to_label()


class QiskitPauli(Pauli):
    """A Qiskit implementation of the Pauli algebra wrapper"""

    from qiskit.quantum_info import Pauli

    def __init__(self, name):
        self.pauli = self.Pauli(name)

    def weight(self):
        return len(self.pauli)

    def to_label(self):
        pauli_nophase = self.Pauli((self.pauli.z, self.pauli.x))
        return pauli_nophase.to_label()

    def basis_change(self, qc):
        circ = qc.copy_empty()
        for p,q in zip(self,qc.qc.qubits):
            if p == QiskitPauli("X"):
                circ.qc.h(q)
            elif p == QiskitPauli("Y"):
                circ.qc.h(q)
                circ.qc.s(q)

        return circ
    
    def weight(self):
        return len(self.pauli)

    def commutes(self,other):
        return self.pauli.commutes(other.pauli)

    def __mul__(self, other):
        result = self.pauli.compose(other.pauli)
        nophase = self.Pauli((result.z, result.x))
        return QiskitPauli(nophase.to_label())

    def __getitem__(self, item):
        return QiskitPauli(self.pauli.__getitem__(item))