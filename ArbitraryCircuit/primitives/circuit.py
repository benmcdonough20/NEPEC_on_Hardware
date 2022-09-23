from abc import ABC, abstractmethod, abstractproperty
from typing_extensions import Self


from .instruction import Instruction, QiskitInstruction
from .term import Pauli

class Circuit(ABC):
    """A class to standardize interface with the native representation of a quantum circuit.
    Implement this class and all the abstact methods to be able to use circuits in any 
    quantum language."""

    @abstractmethod
    def copy_empty(self) -> Self:
        """Return an empty copy of the same circuit with the same qubits/qubit addresses
        such that the new circuit can be composed seamlessly with the old one"""

        pass

    @abstractmethod
    def add_instruction(self, inst : Instruction) -> None:
        """This method takes a representation of an Instruction and adds it to the end 
        of the circuit. The Instruction interface also needs to be implemented"""

        pass

    @abstractmethod
    def add_pauli(self, pauli : Pauli):
        """Append a pauli operator to the circuit"""

        pass

    @abstractmethod
    def barrier(self):
        """Add a compiler directive to maintain either side of the barrier separate"""

        pass

    @abstractmethod
    def measure_all(self, qubits):
        """Add a measure instruction to the desired qubits"""

    @abstractmethod
    def compose(self, other : Self) -> None:
        """Performs the composition of this circuit with other, leaving the ordering and 
        mapping of qubits on both circuits unchanged"""

        pass

    @abstractmethod
    def __getitem__(self) -> Instruction:
        """This method provides an easy way to iterate through all of the instructions in 
        the circuit"""

        pass

    @abstractmethod
    def inverse(self) -> Self:
        """This returns the inverse of the circuit"""

        pass

    @abstractmethod
    def __eq__(self):
        """This method can usually just call the __eq__ method of the underlying circuit"""

        pass

    @abstractmethod
    def num_qubits(self):
        """This returns the number of qubits in the circuit"""

        pass

    @abstractmethod
    def qubits(self):
        """This returns an iterable containing the addresses of qubits in the circuit"""

        pass

    @abstractmethod
    def original(self):
        """Returns the original circuit object in the native language"""
    
        pass

    @property
    @abstractmethod
    def pauli_type(self):
        """Returns the pauli implementation required to interact with the circuit"""
        pass

    def __hash__(self):
        return frozenset([inst for inst in self]).__hash__()


from qiskit import QuantumCircuit
from .term import QiskitPauli

class QiskitCircuit(Circuit):
    """This is an implementation of the Circuit interface for the Qiskit API"""

    def __init__(self, qc : QuantumCircuit):
        self.qc = qc

    def add_instruction(self, inst : QiskitInstruction):
        self.qc.append(inst._instruction)

    def add_pauli(self, pauli : QiskitPauli):
        for q,p in zip(self.qubits(), pauli._pauli):
            self.qc.append(p, [q])

    def barrier(self):
        self.qc.barrier()

    def measure_all(self):
        self.qc.measure_all()

    def compose(self, other : Self):
        self.qc = self.qc.compose(other.qc)
    
    def copy_empty(self):
        return QiskitCircuit(self.qc.copy_empty_like())

    def __getitem__(self, item : int):
        return QiskitInstruction(self.qc.__getitem__(item))

    def __eq__(self, other : Self):
        return self.qc.__eq__(other.qc)

    def inverse(self) -> Self:
        return QiskitCircuit(self.qc.inverse())

    def num_qubits(self):
        return self.qc.num_qubits
    
    def qubits(self):
        return self.qc.qubits

    def original(self):
        return self.qc

    def pauli_type(self):
        return QiskitPauli

    def __hash__(self):
        return super().__hash__()