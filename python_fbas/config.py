"""
Global config object for the python_fbas package.
"""
from typing import Literal
from pysat.solvers import SolverNames

solvers:list[str] = [list(SolverNames.__dict__[s])[::-1][0] for s in SolverNames.__dict__ if not s.startswith('__')]
sat_solver:str = 'cryptominisat5'
card_encoding:Literal['naive','totalizer'] = 'naive'
max_sat_algo:Literal['LRU','RC2'] = 'LRU'
