from pysat.formula import Formula

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

def fixpoint(f, x):
    """Iterate f starting from x until a fixed point is reached."""
    while True:
        y = f(x)
        if x == y:
            return x
        x = y