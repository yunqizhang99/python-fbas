#ifndef MULTI_PARTITIONS_H
#define MULTI_PARTITIONS_H

#include <stddef.h>
#include <stdint.h>

// Function pointer type for the visit function
typedef void (*visit_func_t)(int c[], int v[], int f[], int l);

// Function to print a part of a partition
void print_partition(int c[], int v[], int f[], int l);

// Function to generate and visit all partitions
void generate_partitions(int n[], size_t m, visit_func_t visit);

#endif // MULTI_PARTITIONS_H
