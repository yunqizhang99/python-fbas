import json
import argparse
import importlib
import logging
from python_fbas.sat_based_fbas_analysis import check_intersection, min_splitting_set, min_blocking_set
from python_fbas.overlay import optimal_overlay, constellation_graph
from python_fbas.fbas import FBAS
from python_fbas.fbas_generator import gen_symmetric_fbas

# TODO: add fast mode

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

    # Command for loading validators from Stellarbeat
    parser_stellarbeat = subparsers.add_parser('load-stellarbeat', help="Load validators from Stellarbeat")
    parser_stellarbeat.add_argument('--update', action='store_true', help="Update validators data")

    # Command for checking intersection
    parser_check = subparsers.add_parser('check-intersection', help="Check intersection of quorums")

    # Command for minimum splitting set
    parser_min_split = subparsers.add_parser('min-splitting-set', help="Find minimal splitting set")

    # Command for minimum blocking set
    parser_min_block = subparsers.add_parser('min-blocking-set', help="Find minimal blocking set")

    # Command for optimal overlay
    parser_overlay = subparsers.add_parser('optimal-overlay', help="Find optimal overlay")

    # Command for constellation graph
    parser_constellation = subparsers.add_parser('constellation-overlay', help="Compute constellation overlay")
    # Add encoding option:
    parser_constellation.add_argument('--card-encoding', default="6", help="Specify the cardinality encoding to use (0 to 9; see the pysat documentation)")

    # Command to generate a symmetric fbas
    parser_symmetric = subparsers.add_parser('gen-symmetric-fbas', help="Generate a symmetric FBAS")
    # Add number of validators option:
    parser_symmetric.add_argument('n', type=int, help="Number of validators")
    # Add file output option:
    parser_symmetric.add_argument('--output', help="Output file")


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

    if args.command == 'update-stellarbeat-cache':
        mod = importlib.import_module('python_fbas.stellarbeat_data')
        mod.get_validators(update=True)
        logging.info("Cached data updated with fresh data from Stellarbeat")

    elif args.command == 'check-intersection':
        fbas = _load_fbas()
        result = check_intersection(fbas)
        print(f"Intersection result: {result}")

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

    elif args.command == 'constellation-overlay':
        fbas = _load_fbas()
        # parse card encoding as integer:
        args.card_encoding = int(args.card_encoding)
        result = constellation_graph(fbas, card_encoding=args.card_encoding)
        print(f"Constellation overlay: {result}")

    elif args.command == 'gen-symmetric-fbas':
        gen_symmetric_fbas(args.n, output=args.output)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
