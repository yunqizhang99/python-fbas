# Building

Run `make`.

# Partitions of a multi-set

To print all partitions of a multiset consisting of an element with multiplicity 3 and another element with multiplicity 2, run `./print-multi-partitions 3 2`.

# Searching for an optimal Constellation overlay

Say we have 7 organizations each needing 3 connections (because their qset threshold is 5). Run `./constellation 7 3` to find the best constellation overlay, where the cost is the number of inter-node edges. Run `./constellation --hyper-edges-only 7 3` to find the best constellation overlay, where the cost is the number of hyper edges.

If we have 20 organizations where 15 need 7 connections and 5 need 9, run `./constellation 15 7 5 9`, etc.