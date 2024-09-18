import argparse
from python_fbas.sat_based_fbas_analysis import check_intersection, min_splitting_set, min_blocking_set, optimal_overlay
from python_fbas.fbas import FBAS
from test_utils import get_validators_from_file

def load_fbas_from_file(validators_file):
    validators = get_validators_from_file(validators_file)
    return FBAS.from_stellarbeat_json(validators)

def main():
    parser = argparse.ArgumentParser(description="FBAS analysis CLI")
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # Command for checking intersection
    parser_check = subparsers.add_parser('check-intersection', help="Check intersection of quorums")
    parser_check.add_argument('validators_file', type=str, help="Path to validators JSON file")

    # Command for minimum splitting set
    parser_min_split = subparsers.add_parser('min-splitting-set', help="Find minimal splitting set")
    parser_min_split.add_argument('validators_file', type=str, help="Path to validators JSON file")

    # Command for minimum blocking set
    parser_min_block = subparsers.add_parser('min-blocking-set', help="Find minimal blocking set")
    parser_min_block.add_argument('validators_file', type=str, help="Path to validators JSON file")

    # Command for optimal overlay
    parser_overlay = subparsers.add_parser('optimal-overlay', help="Find optimal overlay")
    parser_overlay.add_argument('validators_file', type=str, help="Path to validators JSON file")

    # Parse arguments
    args = parser.parse_args()

    if args.command == 'check-intersection':
        fbas = load_fbas_from_file(args.validators_file)
        result = check_intersection(fbas)
        print(f"Intersection result: {result}")
    
    elif args.command == 'min-splitting-set':
        fbas = load_fbas_from_file(args.validators_file)
        result = min_splitting_set(fbas)
        print(f"Minimal splitting set: {result}")

    elif args.command == 'min-blocking-set':
        fbas = load_fbas_from_file(args.validators_file)
        result = min_blocking_set(fbas)
        print(f"Minimal blocking set: {result}")

    elif args.command == 'optimal-overlay':
        fbas = load_fbas_from_file(args.validators_file)
        result = optimal_overlay(fbas)
        print(f"Optimal overlay: {result}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
