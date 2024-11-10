#include <stdio.h>
#include <stdlib.h>
#include "multi-partitions.h"

// Global variable to count the number of partitions
uint64_t num_partitions = 0;

// Example visit function
void visit_partition(int c[], int v[], int f[], int l) {
    print_partition(c, v, f, l);
    num_partitions = num_partitions + 1;
}

int main(int argc, char *argv[]) {
    // the user is expected to provide a list of numbers which specifies the multiplicity of each element
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <n1> <n2> ... <nm>\n", argv[0]);
        return 1;
    }
    // now parse the multiset, which is also a vector
    int m = argc - 1;
    int n[m+1]; // to match Knuth's 1-based indexing
    for (int i = 1; i <= m; i++) {
        n[i] = atoi(argv[i]);
    }
    generate_partitions(n, m, visit_partition);
    // print the number of partitions:
    printf("Number of partitions: %lu\n", num_partitions);
    return 0;
}