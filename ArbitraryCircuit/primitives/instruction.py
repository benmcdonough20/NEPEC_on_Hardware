from abc import ABC, abstractmethod

class Instruction(ABC):

    @abstractmethod
    def weight(self):
        pass

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def support(self):
        pass

    @abstractmethod
    def ismeas(self):
        pass

    def __hash__(self):
        return (self.name(),self.support()).__hash__()

    def __eq__(self, other):
        return self.name() == other.name() and self.support() == other.support()


class QiskitInstruction(Instruction):

    def __init__(self, instruction):
        self._instruction = instruction

    def weight(self):
        return self._instruction.operation.num_qubits
    
    def support(self):
        return self._instruction.qubits

    def name(self):
        return self._instruction.operation.name

    def ismeas(self):
        return self.name() == "measure"