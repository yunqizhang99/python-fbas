"""
For compute-clusters, you must first run 'make' in the 'brute-force-search/' directory and then make
sure that the executable 'optimal_cluster_assignment' is in your PATH.
"""

import argparse
import logging
import sys
import subprocess
from python_fbas.constellation.constellation import parse_output

def main():
    parser = argparse.ArgumentParser(description="Constellation CLI")
    # specify log level with --log-level, with default WARNING:
    parser.add_argument('--log-level', default='WARNING', help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    clusters_parser = subparsers.add_parser('compute-clusters', help="compute an assignment of organizations to clusters")
    clusters_parser.add_argument('--thresholds', nargs='+', type=int, required=True, help="List of the form 'm1 t1 m2 t2 ... mn tn' where each ti is a quorum-set threshold and mi is the number of organizations that have this threshold")
    clusters_parser.add_argument('--min-part-size', type=int, default=1, help="Minimum size of a cluster (default: 1)")
    
    args = parser.parse_args()

    if args.command == 'compute-clusters':
        if len(args.thresholds) % 2 != 0 \
            or len(args.thresholds) == 0 \
            or any(not isinstance(args.thresholds[i], int) or args.thresholds[i] <= 0 
                   for i in range(len(args.thresholds))) \
            or any(args.thresholds[1::2].count(args.thresholds[i]) > 1 for i in range(1,len(args.thresholds),2)):
            logging.error("Quorum-set thresholds and their multiplicity must be provided as a list 'm1 t1 m2 t2 ... mn tn' of strictly positive integer with no duplicate thresholds.")
            sys.exit(1)
        # sum all the multiplicities to get the total number of organizations:
        num_orgs = sum(args.thresholds[i] for i in range(0, len(args.thresholds), 2))
        overlay_thresholds = [num_orgs - args.thresholds[i + 1] + 1 for i in range(0, len(args.thresholds), 2)]
        command_args = [args.thresholds[i] if i%2 == 0 else overlay_thresholds[int(i/2)] for i in range(0, len(args.thresholds))] + [args.min_part_size]
        clusters = subprocess.run(['optimal_cluster_assignment'] + [str(x) for x in command_args], capture_output=True, text=True, check=True)
        print(clusters.stdout)
        print(parse_output(clusters.stdout))
    
    # print help:
    elif args.command is None:
        parser.print_help()

if __name__ == "__main__":
    main()