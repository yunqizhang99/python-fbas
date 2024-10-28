#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Generate all partitions of a set using Knuth's Algorithm H

uint64_t N = 0; // number of partitions we found

void print_partition(int a[], int n) {
    for (int i = 1; i <= n; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");
}

void visit_partition(int a[], int n) {
    N = N + 1;
    // print_partition(a, n);
}

void generate_partitions(int n) {
    int a[n+1]; // to match Knuth's 1-based indexing
    int b[n];   // to match Knuth's 1-based indexing
    int m = 1;
    int j;

    // Step H1
    for (int i = 1; i <= n; i++) {
        a[i] = 0;
    }
    for (int i = 1; i <= n-1; i++) {
        b[i] = 1;
    }

    while (1) {
        // Step H2: visit the current restricted-growth string
        h2:
        visit_partition(a, n);
        if (a[n] == m) {
            goto h4;
        }

        // Step H3:
        h3:
        a[n] = a[n] + 1;
        goto h2;

        // Step H4:
        h4:
        j = n - 1;
        while (a[j] == b[j]) {
            j = j - 1;
        }

        // Step H5:
        h5:
        if (j == 1) {
            return;
        }
        a[j] = a[j] + 1;

        // Step H6:
        h6:
        m = b[j] + (a[j] == b[j] ? 1 : 0);
        j = j + 1;
        while (j < n) {
            a[j] = 0;
            b[j] = m;
            j = j + 1;
        }
        a[n] = 0;
        goto h2;
    }
}

// otain n, the cardinality of the set, as first argument:
int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s n\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    generate_partitions(n);
    // print the number of partitions found:
    printf("%lu\n", N);
    return 0;
}
