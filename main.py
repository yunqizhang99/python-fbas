import json
import argparse
import importlib
import logging
from python_fbas.sat_based_fbas_analysis import check_intersection, min_splitting_set, min_blocking_set, is_in_min_quorum_of
from python_fbas.overlay import optimal_overlay
from python_fbas.fbas import FBAS
from python_fbas.fbas_generator import gen_symmetric_fbas
from python_fbas.fbas_graph import FBASGraph
from python_fbas.fbas_graph_analysis import find_disjoint_quorums, find_disjoint_quorums_using_pysat_fmla, find_minimal_splitting_set
from python_fbas.z3_fbas_graph_analysis import find_disjoint_quorums as z3_find_disjoint_quorums

def load_json_from_file(validators_file):
    with open(validators_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def _load_json_from_stellarbeat() -> list[dict]:
    # load dynamically because this triggers fetching data from Stellarbeat
    mod = importlib.import_module('python_fbas.stellarbeat_data')
    return mod.get_validators()

def _load_fbas(args) -> FBAS:
    if args.fbas == 'stellarbeat':
        fbas = FBAS.from_json(_load_json_from_stellarbeat())
        logging.info("Validators loaded from Stellarbeat")
        logging.info("Sanitizing")
        fbas = fbas.sanitize()
        logging.info("Sanitized fbas has %d validators", len(fbas.qset_map))
        return fbas
    else:
        return FBAS.from_json(load_json_from_file(args.fbas))
    
def _load_fbas_graph(args) -> FBASGraph:
    if args.fbas == 'stellarbeat':
        return FBASGraph.from_json(_load_json_from_stellarbeat())
    else:
        return FBASGraph.from_json(load_json_from_file(args.fbas))

def main():
    parser = argparse.ArgumentParser(description="FBAS analysis CLI")
    # specify log level with --log-level, with default WARNING:
    parser.add_argument('--log-level', default='WARNING', help="Logging level")
    
    # specify a data source:
    parser.add_argument('--fbas', default='stellarbeat', help="Where to find the description of the FBAS to analyze")
    parser.add_argument('--validator', default=None, help="Public key of the validator we are taking the viewpoint of")


    parser.add_argument('--new', action='store_true', help="Whether to use the new codebase")

    # specify whether to group validators by some metadata field:
    parser.add_argument('--group-by', default=None, help="Group validators using the provided metadata field (e.g. 'homeDomain')")

    # subcommands:
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # Command for updating the data from Stellarbeat
    parser_stellarbeat = subparsers.add_parser('update-stellarbeat-cache', help="Update data downloaded from stellarbeat.io")

    # Command for checking intersection
    parser_check_intersection = subparsers.add_parser('check-intersection', help="Check intersection of quorums")
    # add --fast option to check-intersection:
    parser_check_intersection.add_argument('--fast', action='store_true', help="Use the fast heuristic")
    parser_check_intersection.add_argument('--pysat_fmla', action='store_true', help="Build pysat Formlas (slow)")
    parser_check_intersection.add_argument('--cnf', action='store_true', help="Directly create a CNF constraint")
    parser_check_intersection.add_argument('--z3', action='store_true', help="Use Z3 instead of pysat")
    parser_check_intersection.add_argument('--flatten', action='store_true', help="Flatten the graph before checking intersection")

    # Command for minimum splitting set
    parser_min_split = subparsers.add_parser('min-splitting-set', help="Find minimal splitting set")

    # Command for minimum blocking set
    parser_min_block = subparsers.add_parser('min-blocking-set', help="Find minimal blocking set")

    # Command for optimal overlay
    parser_overlay = subparsers.add_parser('optimal-overlay', help="Find optimal overlay")

    # Command to generate a symmetric fbas
    parser_symmetric = subparsers.add_parser('gen-symmetric-fbas', help="Generate a symmetric FBAS")
    # Add number of validators option:
    parser_symmetric.add_argument('n', type=int, help="Number of validators")
    # Add file output option:
    parser_symmetric.add_argument('--output', help="Output file")

    # Command taking two validators and checking if one is in the min quorum of the other:
    parser_is_in_min_quorum_of = subparsers.add_parser('is-in-min-quorum-of', help="Check if one validator is in the min quorum of another")
    parser_is_in_min_quorum_of.add_argument('validator1', help="Public key of the first validator")
    parser_is_in_min_quorum_of.add_argument('validator2', help="Public key of the second validator")

    # Parse arguments
    args = parser.parse_args()

    # set the log level:
    logging.getLogger().setLevel(args.log_level)

    # group-by only applies to min-splitting-set:
    if (args.command != 'min-splitting-set' or args.new) and args.group_by:
        logging.error("--group-by only applies to the 'min-splitting-set' command and is not supported with the new codebase")
        exit(1)

    if args.command == 'update-stellarbeat-cache':
        mod = importlib.import_module('python_fbas.stellarbeat_data')
        mod.get_validators(update=True)
        logging.info("Cached data updated with fresh data from Stellarbeat")
        exit(0)

    if args.new:
        logging.info("Using the new codebase (fbas_graph.py)")
        fbas = _load_fbas_graph(args)
        if args.validator:
            fbas = fbas.restrict_to_reachable(args.validator)
        if args.command == 'check-intersection':
            # fast, z3, cnf, and pysat are mutually exclusive:
            if sum([args.fast, args.z3, args.cnf, args.pysat_fmla]) > 1:
                logging.error("--fast, --z3, --cnf, and --pysat-fmla are mutually exclusive")
                exit(1)
            if args.fast:
                result = fbas.fast_intersection_check()
                print(f"Intersection-check result: {result}")
            elif args.z3:
                result = z3_find_disjoint_quorums(fbas)
                print(f"Disjoint quorums: {result}")
            elif args.cnf:
                result = find_disjoint_quorums(fbas, flatten=args.flatten)
                print(f"Disjoint quorums: {result}")
            elif args.pysat_fmla:
                result = find_disjoint_quorums_using_pysat_fmla(fbas, flatten=args.flatten)
                print(f"Disjoint quorums: {result}")
            else:
                logging.error("Must specify one of --fast, --z3, --cnf, or --pysat-fmla")
                exit(1)
        elif args.command == 'min-splitting-set':
            result = find_minimal_splitting_set(fbas)
            print(f"Minimal splitting set: {result}")
        else:
            print("Command not supported with the new codebase")
            parser.print_help()
            exit(1)
    else:
        fbas = _load_fbas(args)
        if args.command == 'check-intersection':
            # --fast require --validator:
            if args.fast and (not args.validator or args.new or args.flatten):
                logging.error("--fast requires --validator")
                exit(1)
            if not args.fast:
                result = check_intersection(fbas)
                print(f"Intersection-check result: {result}")
            else:
                result = fbas.fast_intersection_check(args.validator)
                print(f"Intersection-check result: {result}")

        elif args.command == 'min-splitting-set':
            result = min_splitting_set(fbas, group_by=args.group_by)

        elif args.command == 'min-blocking-set':
            result = min_blocking_set(fbas)
            print(f"Minimal blocking set: {result}")

        elif args.command == 'optimal-overlay':
            result = optimal_overlay(fbas)
            print(f"Optimal overlay: {result}")

        elif args.command == 'gen-symmetric-fbas':
            gen_symmetric_fbas(args.n, output=args.output)

        elif args.command == 'is-in-min-quorum-of':
            result = is_in_min_quorum_of(fbas, args.validator1, args.validator2)
            print(f"Is {args.validator1} ({fbas.metadata[args.validator1]['name']}) in a min quorum of {args.validator2} ({fbas.metadata[args.validator2]['name']})? {result}")

        else:
            parser.print_help()

if __name__ == "__main__":
    main()
