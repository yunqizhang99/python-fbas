"""
For compute-clusters, you must first run 'make' in the 'brute-force-search/' directory and then make
sure that the executable 'optimal_cluster_assignment' is in your PATH.
"""

import argparse
import logging
import os
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Constellation CLI")
    # specify log level with --log-level, with default WARNING:
    parser.add_argument('--log-level', default='WARNING', help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    # subcommands:
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # Command for updating the data from Stellarbeat
    clusters_parser = subparsers.add_parser('compute-clusters', help="compute an assignment of organizations to clusters")
    clusters_parser.add_argument('--thresholds', nargs='+', type=int, required=True, help="List of the form 'm1 t1 m2 t2 ... mn tn' where each ti is a quorum-set threshold and mi is the number of organizations that have this threshold")
    
    args = parser.parse_args()

    if args.command == 'compute-clusters':
        if len(args.thresholds) % 2 != 0 or len(args.thresholds) == 0:
            logging.error("Quorum-set thresholds and their multiplicity must be provided as a list 'm1 t1 m2 t2 ... mn tn'")
            sys.exit(1)
        # sum all the multiplicities to get the total number of organizations:
        num_orgs = sum(args.thresholds[i] for i in range(0, len(args.thresholds), 2))
        overlay_thresholds = [num_orgs - args.thresholds[i + 1] + 1 for i in range(0, len(args.thresholds), 2)]
        command_args = [args.thresholds[i] if i%2 == 0 else overlay_thresholds[int(i/2)] for i in range(0, len(args.thresholds))]
        clusters = subprocess.run(['optimal_cluster_assignment'] + [str(x) for x in command_args], capture_output=True, text=True, check=True)
        print(clusters.stdout)

if __name__ == "__main__":
    main()