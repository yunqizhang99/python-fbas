import json
import argparse
import importlib
import logging
from python_fbas.sat_based_fbas_analysis import check_intersection, min_splitting_set, min_blocking_set, is_in_min_quorum_of
from python_fbas.overlay import optimal_overlay
from python_fbas.fbas import FBAS
from python_fbas.fbas_generator import gen_symmetric_fbas

# TODO: add fast mode
# TODO: a --viewpoint (everything is subjective in a FBAS!)

def load_fbas_from_file(validators_file):
    with open(validators_file, 'r', encoding='utf-8') as f:
        validators = json.load(f)
    return FBAS.from_json(validators)

def main():
    parser = argparse.ArgumentParser(description="FBAS analysis CLI")
    # specify log level with --log-level, with default WARNING:
    parser.add_argument('--log-level', default='WARNING', help="Logging level")
    
    # specify a data source:
    parser.add_argument('--fbas', default='stellarbeat', help="Where to find the description of the FBAS to analyze")

    # specify whether to group validators by some metadata field:
    # TODO allow specifying a nested attribute
    parser.add_argument('--group-by', default=None, help="Group validators using the provided metadata field (e.g. 'homeDomain')")

    # subcommands:
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # Command for updating the data from Stellarbeat
    parser_stellarbeat = subparsers.add_parser('update-stellarbeat-cache', help="Update data downloaded from stellarbeat.io")

    # Command for checking intersection
    parser_check_intersection = subparsers.add_parser('check-intersection', help="Check intersection of quorums")
    # add --fast option to check-intersection:
    parser_check_intersection.add_argument('--fast', action='store_true', help="Use the fast heuristic")
    parser_check_intersection.add_argument('--validator', help="Public key of the validator from whose viewpoint to check intersection")

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

    def _load_fbas_from_stellarbeat():
        mod = importlib.import_module('python_fbas.stellarbeat_data')
        mod.get_validators()
        fbas = FBAS.from_json(mod.validators)
        logging.info("Validators loaded from Stellarbeat")
        logging.info("Sanitizing")
        fbas = fbas.sanitize()
        logging.info("Sanitized fbas has %d validators", len(fbas.qset_map))
        return fbas

    def _load_fbas():
        if args.fbas == 'stellarbeat':
            return _load_fbas_from_stellarbeat()
        else:
            return load_fbas_from_file(args.fbas)

    # Parse arguments
    args = parser.parse_args()

    # set the log level:
    logging.getLogger().setLevel(args.log_level)

    # group-by only applies to min-splitting-set:
    if args.command != 'min-splitting-set' and args.group_by:
        logging.error("--group-by only applies to the 'min-splitting-set' command")
        exit(1)

    if args.command == 'update-stellarbeat-cache':
        mod = importlib.import_module('python_fbas.stellarbeat_data')
        mod.get_validators(update=True)
        logging.info("Cached data updated with fresh data from Stellarbeat")

    elif args.command == 'check-intersection':
        # --fast require --validator:
        if args.fast and not args.validator:
            logging.error("--fast requires --validator")
            exit(1)
        fbas = _load_fbas()
        if not args.fast:
            result = check_intersection(fbas)
            print(f"Intersection-check result: {result}")
        else:
            result = fbas.fast_intersection_check(args.validator)
            print(f"Intersection-check result: {result}")

    elif args.command == 'min-splitting-set':
        fbas = _load_fbas()
        result = min_splitting_set(fbas, group_by=args.group_by)

    elif args.command == 'min-blocking-set':
        fbas = _load_fbas()
        result = min_blocking_set(fbas)
        print(f"Minimal blocking set: {result}")

    elif args.command == 'optimal-overlay':
        fbas = _load_fbas()
        result = optimal_overlay(fbas)
        print(f"Optimal overlay: {result}")

    elif args.command == 'gen-symmetric-fbas':
        gen_symmetric_fbas(args.n, output=args.output)

    elif args.command == 'is-in-min-quorum-of':
        fbas = _load_fbas()
        result = is_in_min_quorum_of(fbas, args.validator1, args.validator2)
        print(f"Is {args.validator1} ({fbas.metadata[args.validator1]['name']}) in a min quorum of {args.validator2} ({fbas.metadata[args.validator2]['name']})? {result}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
