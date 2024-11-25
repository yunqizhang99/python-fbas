from collections.abc import Sequence
from pysat.formula import Formula

def powerset(s : Sequence):
    """A generator for the powerset of s. Assume elements in s are unique."""
    x = len(s)
    # each x-bit number represents a subset of s:
    for i in range(1 << x):
        yield {s[j] for j in range(x) if (i & (1 << j))}

# pysat stuff

def to_cnf(fmlas : list[Formula]) -> list:
    """Convert a list of formulas to CNF."""
    # NOTE: iterating through f first triggers clausification
    return [c for f in fmlas for c in f]

def clause_of_pseudo_atom(a: Formula) -> list:
    """a must be of the form Neg(Atom(...)) or Atom(...); returns the corresponding clause"""
    a.clausify()
    return a.clauses[0]

def vars_of_cnf(cnf: list[list[int]]) -> set[int]:
    """Extract the variables from a CNF."""
    return {abs(l) for c in cnf for l in c}
