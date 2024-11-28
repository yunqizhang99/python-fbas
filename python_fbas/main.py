"""
Main CLI for the FBAS analysis tool
"""

import json
import argparse
import logging
from python_fbas.fbas_graph import FBASGraph
from python_fbas.fbas_graph_analysis import find_disjoint_quorums_, find_disjoint_quorums, \
    find_disjoint_quorums_using_pysat_fmla, find_minimal_splitting_set
from python_fbas.stellarbeat_data import get_validators as get_stellarbeat_validators
import python_fbas.config as config

def _load_json_from_file(validators_file):
    with open(validators_file, 'r', encoding='utf-8') as f:
        return json.load(f)

    
def _load_fbas_graph(args) -> FBASGraph:
    if args.fbas == 'stellarbeat':
        return FBASGraph.from_json(get_stellarbeat_validators())
    return FBASGraph.from_json(_load_json_from_file(args.fbas))

def main():
    parser = argparse.ArgumentParser(description="FBAS analysis CLI")
    # specify log level with --log-level, with default WARNING:
    parser.add_argument('--log-level', default='INFO', help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    
    # specify a data source:
    parser.add_argument('--fbas', default='stellarbeat', help="Where to find the description of the FBAS to analyze (must be 'stellarbeat' or a path to a JSON file)")
    parser.add_argument('--validator', default=None, help="Restrict the FBAS to what's reachable from this validator")

    parser.add_argument('--encoding', default='cnf', help="Encode the SAT problem in CNF (--cnf) or pysat Formulas (--pysat-fmla). The pysat Formula encoding is slow and mostly there for testing and didactic purposes)")
    parser.add_argument('--heuristic-first', action='store_true', help="When available, first try a fast, sound but incomplete heuristic")

    parser.add_argument('--cardinality-encoding', default='totalizer', help="Cardinality encoding, either 'naive' or 'totalizer'")
    parser.add_argument('--sat-solver', default='cryptominisat5', help=f"SAT solver to use ({config.solvers}). See the documentation of the pysat package for more information.")
    parser.add_argument('--max-sat-algo', default='LRU', help="MaxSAT algorithm to use (LRU or RC2)")
    parser.add_argument('--output-problem', default=None, help="Write the constraint-satisfaction problem to the provided path")

    # subcommands:
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # Command for updating the data from Stellarbeat
    subparsers.add_parser('update-stellarbeat-cache', help="Update data downloaded from stellarbeat.io")

    # Command for checking intersection
    parser_is_intertwined = subparsers.add_parser('check-intersection', help="Check that the FBAS is intertwined (i.e. whether all quorums intersect)")
    parser_is_intertwined.add_argument('--fast', action='store_true', help="Use the fast heuristic (which does not use a SAT solver and only returns true, meaning all quorums intersect, or unknown)")

    # Command for minimum splitting set
    subparsers.add_parser('min-splitting-set', help="Find minimal-cardinality splitting set")
    # subparsers.add_parser('min-blocking-set', help="Find minimal-cardinality blocking set")

    # subparsers.add_parser('intersection-check-to-dimacs', help="Create a DIMACS file containing a problem that is satisfiable if and only there are disjoint quorums")
    
    args = parser.parse_args()

    # set config:

    if args.encoding not in ['cnf', 'pysat-fmla']:
        logging.error("Encoding must be either 'cnf' or 'pysat-fmla'")
        exit(1)
    config.heuristic_first = args.heuristic_first
    config.output = args.output_problem
    if args.cardinality_encoding not in ['naive', 'totalizer']:
        logging.error("Cardinality encoding must be either 'naive' or 'totalizer'")
        exit(1)
    if args.cardinality_encoding == 'totalizer' and args.encoding == 'pysat-fmla':
        logging.error("Totalizer encoding is not supported with --pysat-fmla")
        exit(1)
    config.card_encoding = args.cardinality_encoding

    if args.sat_solver not in config.solvers:
        logging.error("Solver must be one of %s", config.solvers)
        exit(1)
    config.sat_solver = args.sat_solver

    if args.max_sat_algo not in ['LRU', 'RC2']:
        logging.error("MaxSAT algorithm must be either 'LRU' or 'RC2'")
        exit(1)
    config.max_sat_algo = args.max_sat_algo

    debug_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if args.log_level not in debug_levels:
        logging.error("Log level must be one of %s", debug_levels)
        exit(1)
    logging.getLogger().setLevel(args.log_level)

    # Run commands:

    if args.command == 'update-stellarbeat-cache':
        get_stellarbeat_validators(update=True)
        exit(0)

    fbas = _load_fbas_graph(args)
    if args.validator:
        fbas = fbas.restrict_to_reachable(args.validator)
    if args.command == 'check-intersection':
        if args.fast:
            result = fbas.fast_intersection_check()
            print(f"Intersection-check result: {result}")
            exit(0)
        elif args.encoding == 'cnf':
            result = find_disjoint_quorums_(fbas)
            print(f"Disjoint quorums: {result}")
            exit(0)
        elif args.encoding == 'pysat-fmla':
            result = find_disjoint_quorums_using_pysat_fmla(fbas)
            print(f"Disjoint quorums: {result}")
            exit(0)
        else:
            logging.error("Must specify one of --fast, --cnf, or --pysat-fmla")
            exit(1)
    elif args.command == 'min-splitting-set':
        if args.encoding == 'pysat-fmla':
            logging.error("min-splitting-set is not supported with --pysat-fmla")
            exit(1)
        result = find_minimal_splitting_set(fbas)
        print(f"Minimal splitting-set cardinality is: {len(result[0])}")
        print(f"Example:\n{result[0]}\nsplits quorums\n{result[1]}\nand\n{result[2]}")
        exit(0)
    else:
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()