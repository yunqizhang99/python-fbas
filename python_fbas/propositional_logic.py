"""
This module provides a simple implementation of propositional logic formulas, including cardinality
constraints. The goal is do avoid all the bookkeeping done by pysat, which makes things too slow.

The main functionality is conversion to CNF. This is implemented using the Tseitin transformation.
We are not expecting formulas to share subformulas, so we will not keep track of which variables
correspond to which subformulas. By convention, the last clause in the CNF is a unit clause with the
variable corresponding to the formula itself.
"""

from abc import ABC, abstractmethod
from itertools import combinations
from typing import Any
import python_fbas.config as config

Clauses = list[list[int]]

next_int:int = 1 # global variable to generate unique variable names
variables:dict[Any,int] = {} # maps variable names to integers
variables_inv:dict[int,Any] = {} # inverse of variables; maps integers to variable names

def var(v:Any) -> int:
    """
    Get the integer corresponding to a variable name.
    """
    global next_int
    if v not in variables:
        variables[v] = next_int
        variables_inv[next_int] = v
        next_int += 1
    return variables[v]

def anonymous_var() -> int:
    """
    Get the next integer variable; do not associate it with a name.
    """
    global next_int
    next_int += 1
    return next_int - 1

class Formula(ABC):
    """
    Abstract base class for propositional logic formulas.
    """
    @abstractmethod
    def to_cnf(self) -> Clauses:
        """
        Convert the formula to CNF. This will be a very basic application of the Tseitin
        transformation. We are not expecting formulas to share subformulas, so we will not keep
        track of which variables correspond to which subformulas. By convention, the last clause in
        the CNF is a unit clause with the variable corresponding to the formula itself.
        """
        pass

def to_cnf(fmlas: list[Formula]) -> Clauses:
    """
    Convert a list of formulas to CNF.
    """
    return [c for f in fmlas for c in f.to_cnf()]

class Atom(Formula):
    """
    A propositional logic atom.
    """
    def __init__(self, identifier: Any):
        self.identifier = identifier

    def __str__(self):
        return self.identifier

    def to_cnf(self) -> Clauses:
        return [[var(self.identifier)]]

class Not(Formula):
    """
    Negation.
    """
    def __init__(self, operand: Formula):
        self.operand = operand

    def __str__(self):
        return f'~({self.operand})'

    def to_cnf(self) -> Clauses:
        v = anonymous_var()
        op_clauses = self.operand.to_cnf()
        op_atom = op_clauses[-1][0] # that's the variable corresponding to the operand
        assert op_atom > 0
        new_clauses = [[-v, -op_atom],[v, op_atom]]
        return op_clauses[:-1] + new_clauses + [[v]]

class And(Formula):
    """
    Conjunction.
    """
    def __init__(self, *operands: Formula):
        self.operands = operands

    def __str__(self):
        return ' & '.join(f'({str(op)})' for op in self.operands)
    
    def to_cnf(self) -> Clauses:
        v = anonymous_var()
        if not self.operands:
            return [[v]] # trivially satisfiable
        ops_clauses = [op.to_cnf() for op in self.operands]
        ops_atoms = [c[-1][0] for c in ops_clauses]
        assert all(op > 0 for op in ops_atoms)
        new_clauses = [[-a for a in ops_atoms] + [v]] + [[-v, a] for a in ops_atoms]
        return [c for cs in ops_clauses for c in cs[:-1]] + new_clauses + [[v]]

class Or(Formula):
    """
    Disjunction.
    """
    def __init__(self, *operands: Formula):
        self.operands = operands

    def __str__(self):
        return ' | '.join(f'({str(op)})' for op in self.operands)
    
    def to_cnf(self) -> Clauses:
        v = anonymous_var()
        if not self.operands:
            return [[-v],[v]] # unsatisfiable
        ops_clauses = [op.to_cnf() for op in self.operands]
        ops_atoms = [c[-1][0] for c in ops_clauses]
        assert all(op > 0 for op in ops_atoms)
        new_clauses = [[-a, v] for a in ops_atoms] + [[-v] + ops_atoms]
        return [c for cs in ops_clauses for c in cs[:-1]] + new_clauses + [[v]]
    
class Implies(Formula):
    """
    Implication. The last operand is the conclusion.
    """
    def __init__(self, *operands: Formula):
        assert len(operands) >= 2
        self.operands = operands

    def __str__(self):
        return '& '.join(f'({str(op)})' for op in self.operands[:-1]) + ' -> ' + str(self.operands[-1])
    
    def to_cnf(self) -> Clauses:
        return Or(Not(And(*self.operands[:-1])), self.operands[-1]).to_cnf()

class Card(Formula):
    """
    A cardinality constraint. Only supports atoms.
    """
    def __init__(self, threshold: int, *operands: Atom):
        assert threshold > 0 and len(operands) >= threshold
        self.threshold = threshold
        self.operands = operands

    def __str__(self):
        return f'{self.threshold} out of {self.operands}'

    def to_cnf(self) -> Clauses:
        if config.card_encoding == 'naive':
            fmla = Or(*[And(*c) for c in combinations(self.operands, self.threshold)])
            return fmla.to_cnf()
        else:
            raise NotImplementedError
