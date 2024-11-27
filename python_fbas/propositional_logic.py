"""
This module provides a simple implementation of propositional logic formulas, including cardinality
constraints. The goal is do avoid all the bookkeeping done by pysat, which makes things too slow.
The main functionality is conversion to CNF.
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod

Clauses = list[list[int]]

next_int:int = 1 # global variable to generate unique variable names
variables:dict[str,int] = {} # maps variable names to integers
variables_inv:dict[int,str] = {} # inverse of variables; maps integers to variable names

def var(name:str) -> int:
    """
    Get the integer corresponding to a variable name.
    """
    global next_int
    if name not in variables:
        variables[name] = next_int
        variables_inv[next_int] = name
        next_int += 1
    return variables[name]

class Formula(ABC):
    """
    Abstract base class for propositional logic formulas.
    """
    @abstractmethod
    def to_cnf(self) -> Clauses:
        """
        Convert the formula to CNF. This will be a very basic application of the Tseitin
        transformation. We are not expecting formulas to share subformulas, so we will not keep
        track of which variables correspond to which subformulas.
        """
        pass

class Atom(Formula):
    """
    A propositional logic atom.
    """
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def to_cnf(self) -> Clauses:
        return [[var(self.name)]]

class Not(Formula):
    """
    Negation.
    """
    def __init__(self, operand: Formula):
        self.operand = operand

    def __str__(self):
        return f'~({self.operand})'

    def to_cnf(self) -> Clauses:
        raise NotImplementedError

class And(Formula):
    """
    Conjunction.
    """
    def __init__(self, *operands: Formula):
        self.operands = operands

    def __str__(self):
        return ' & '.join(f'({str(op)})' for op in self.operands)
    
    def to_cnf(self) -> Clauses:
        raise NotImplementedError

class Or(Formula):
    """
    Disjunction.
    """
    def __init__(self, *operands: Formula):
        self.operands = operands

    def __str__(self):
        return ' | '.join(f'({str(op)})' for op in self.operands)
    
    def to_cnf(self) -> Clauses:
        raise NotImplementedError
    
class Implies(Formula):
    """
    Implication.
    """
    def __init__(self, *operands: Formula):
        self.operands = operands

    def __str__(self):
        return '& '.join(f'({str(op)})' for op in self.operands[:-1]) + ' -> ' + str(self.operands[-1])
    
    def to_cnf(self) -> Clauses:
        raise NotImplementedError

class Card(Formula):
    """
    A cardinality constraint.
    """
    def __init__(self, lower: int, *operands: Formula):
        self.lower = lower
        self.operands = operands

    def __str__(self):
        return f'{self.lower} out of {self.operands}'

    def to_cnf(self) -> Clauses:
        raise NotImplementedError