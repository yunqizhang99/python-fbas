#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "multi-partitions.h"

#define min(a, b) ((a) < (b) ? (a) : (b))

// Generate all partitions of a vector using Knuth's Algorithm M

// We represent a partition of a vector using three arrays c, v, and f.
// c denotes component numbers, v denotes component values, and f is the "stack frame" array.
// The part number l of the partition consists of the elements f[l] through f[l+1]-1.
// For example, let's say n=2. If f[2]=2, f[3]=4, c[2]=1, v[2]=3, c[3]=2, and v[3]=4, then the second partition is the vector [3, 4].
// If some components are missing, they are 0 by default.
// For example, if f[2]=2, f[3]=3, c[2]=2, and v[2]=3, then the second partition is the vector [0, 3].

// Function to print part k of a partition:
void print_part(int c[], int v[], int f[], int k) {
    for (int i = f[k]; i < f[k+1]; i++) {
        for (int n = 0; n < v[i]; n++)
            printf("%d", c[i]);
    }
}

// Function to print a partition:
void print_partition(int c[], int v[], int f[], int l) {
    for (int k = 0; k < l+1; k++) {
        print_part(c, v, f, k);
        if (k == l)
            break;
        printf("|");
    }
    printf("\n");
}

size_t part_size(int c[], int v[], int f[], int k) {
    size_t size = 0;
    for (int i = f[k]; i < f[k+1]; i++) {
        size += v[i];
    }
    return size;
}

// Global variable to count the number of partitions found
uint64_t num_partitions = 0;

// Example visit function
void example_visit_partition(int c[], int v[], int f[], int l) {
    print_partition(c, v, f, l);
    num_partitions = num_partitions + 1;
}

// we have a vector n[1],n[2],...,n[m]
// equivalently, we have m elements, where each element 0 < i <= m has multiplicity n[i]
void generate_partitions(int n[], size_t m, visit_func_t visit, size_t min_size) {

    // number of elements:
    size_t N = 0;
    for (size_t i = 1; i <= m; i++) {
        N += n[i];
    }

    // Step M1: initialization
    int c[N*m]; // component numbers
    int u[N*m]; // yet-unpartitioned amount
    int v[N*m]; // c component of the current part
    int f[N*m+1]; // stack frames
    int l = 0; // current part
    int a = 0;
    int b = m; // current stack frame runs from a to b-1:
    int j;
    int k;
    int x;
    f[0] = 0;
    f[1] = m;
    for (int j = 0; j < m; j++) {
        c[j] = j + 1;
        u[j] = n[j+1];
        v[j] = n[j+1];
    }

    // Step M2
    m2:
    j = a;
    k = b;
    x = 0;
    while (j < b) {
        u[k] = u[j] - v[j];
        if (u[k] == 0) {
            x = 1;
            j = j + 1;
        }
        else if (x == 0) {
            c[k] = c[j];
            v[k] = min(v[j], u[k]);
            x = u[k] < v[j] ? 1 : 0;
            k = k + 1;
            j = j + 1;
        }
        else {
            c[k] = c[j];
            v[k] = u[k];
            k = k + 1;
            j = j + 1;
        }
    }
    // TODO if u is too small we should decrease v and restart in the same frame

    // Step M3
    m3:
    if (k > b) {
        a = b;
        b = k;
        l = l + 1;
        f[l+1] = b;
        goto m2;
    }

    // Step M4
    m4:
    visit(c, v, f, l);

    // Step M5
    m5:
    j = b - 1;
    while (v[j] == 0) {
        j = j - 1;
    }
    if (j == a && v[j] == 1) {
        goto m6;
    }
    else {
        v[j] = v[j] - 1;
        for (int k = j+1; k < b; k++) {
            v[k] = u[k];
        }
        // backtrack if size limit reached:
        if (part_size(c, v, f, l) < min_size) {
            goto m6;
        }
        goto m2;
    }

    // Step M6
    m6:
    if (l == 0) {
        return;
    }
    else {
        l = l - 1;
        b = a;
        a = f[l];
        goto m5;
    }
}

/* int main(int argc, char *argv[]) { */
    /* // the user is expected to provide a list of numbers which specifies the multiplicity of each element */
    /* if (argc < 2) { */
        /* fprintf(stderr, "Usage: %s <n1> <n2> ... <nm>\n", argv[0]); */
        /* return 1; */
    /* } */
    /* // now parse the multiset, which is also a vector */
    /* int m = argc - 1; */
    /* int n[m+1]; // to match Knuth's 1-based indexing */
    /* for (int i = 1; i <= m; i++) { */
        /* n[i] = atoi(argv[i]); */
    /* } */
    /* generate_partitions(n, m, example_visit_partition, 1); */
    /* // print the number of partitions: */
    /* printf("Number of partitions: %lu\n", num_partitions); */
    /* return 0; */
/* } */
