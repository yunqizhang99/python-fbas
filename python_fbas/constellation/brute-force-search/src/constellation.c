#include <stdio.h>
#include <stdlib.h> // Include for atoi
#include <limits.h> // Include for INT_MAX
#include "multi-partitions.h"

#define max(a, b) ((a) < (b) ? (b) : (a))

// array of thresholds, which must be set before calling cost
// we assume that the components are sorted in decreasing threshold order (i.e. t[1] >= t[2] >= ...)
int *t;

// Given a partition, compute its cost, i.e. the number of edges in the resulting Constellation graph
// If the thresholds are not satisfied, the cost is INT_MAX
// See multi-partitions.c for comments about what c, v, and f are
int cost(int c[], int v[], int f[], int l) {
    // first compute the cardinality of each part:
    int card[l+1];
    for (int k = 0; k < l+1; k++) {
        card[k] = 0;
        for (int i = f[k]; i < f[k+1]; i++)
            card[k] += v[i];
    }
    // next we check if each organization has sufficiently many hyper-edges (more than its threshold), and we return INT_MAX if any does not
    for (int k = 0; k < l+1; k++) {
        // compute number of hyper-edges of each organization in the part k
        // we start with the intra-part hyper-edges
        int e[card[k]];
        // first count the intra-part hyper-edges:
        for (int i = 0; i < card[k]; i++)
            e[i] = card[k]; // not card[k]-1 because "self" counts as a hyper-edge
        // next count the inter-part hyper-edges:
        for (int j = 0; j < l+1; j++) {
            if (j == k) continue;
            if (card[j] < card[k]) {
                for (int i = 0; i < card[k]; i++)
                    e[i] += 1;
            }
            else {
                // the first card[j] % card[k] organizations get one more hyper-edge
                for (int i = 0; i < card[j] % card[k]; i++)
                    e[i] += (card[j] / card[k]) + 1;
                // the rest get card[j] / card[k] hyper-edges
                for (int i = card[j] % card[k]; i < card[k]; i++)
                    e[i] += card[j] / card[k];
            }
        }
        // finally, check that each organization has at least its threshold number of hyper-edges
        // iterate in the frame:
        int j = 0;
        for (int i = f[k]; i < f[k+1]; i++) {
            int threshold = t[c[i]];
            int num_orgs = v[i];
            for (int n = 0; n < num_orgs; n++) {
                if (e[j] < threshold)
                    return INT_MAX;
                j++;
            }
        }
    }
    // next (if we have not returned INT_MAX), compute the cost of the partition
    int sum = 0;
    // sum all intra-part hyper-edges:
    for (int k = 0; k < l+1; k++)
        // 6 * (the edge cardinality of the complete graph with card vertices) + 3*card:
        sum += 3 * card[k] * (card[k] - 1) + 3 * card[k];
        // sum += (card[k] * (card[k] - 1)) / 2; // if we only count hyper-edges
    // finally, iterate over all unordered pairs of parts and sum the inter-part hyper-edges (each cost 9):
    for (int i = 0; i < l; i++)
        for (int j = i+1; j < l+1; j++)
            sum += 9*max(card[i], card[j]);
            /* sum += max(card[i], card[j]); // if we only count hyper-edges */
    return sum;
}

int best_cost = INT_MAX;
int invalid = 0;
void visit_partition(int c[], int v[], int f[], int l) {
    int cst = cost(c, v, f, l);
    if (cst == INT_MAX)
        invalid++;
    // print_partition(c, v, f, l);
    // printf("Cost: %d\n", cst);
    if (cst < best_cost) {
        best_cost = cst;
        print_partition(c, v, f, l);
        printf("Cost: %d\n", cst);
    }
}

int main(int argc, char *argv[]) {
    // the user is expected to provide a list of pairs of numbers whose first element specifies a multiplicity and whose second element specifies a threshold value.
    // for example, 3 2 4 3 specifies that we have 3 organization with threshold 2 and 4 organizations with threshold 3
    if (argc % 2 != 1 || argc < 3) {
        printf("Usage: %s m1 t1 m2 t2 ...\n", argv[0]);
        return 1;
    }
    size_t n = (argc - 1) / 2;
    int m[n+1];
    // update t to point to a size-(n+1) array
    t = (int *) malloc((n+1) * sizeof(int));
    for (size_t i = 1; i <= n; i++) { // one-based indexing
        m[i] = atoi(argv[1 + 2*(i-1)]);
        t[i] = atoi(argv[2 + 2*(i-1)]);
    }
    // Now sort the components in decreasing threshold order
    for (size_t i = 1; i < n; i++) {
        for (size_t j = i+1; j <= n; j++) {
            if (t[j] > t[i]) {
                int tmp = m[i];
                m[i] = m[j];
                m[j] = tmp;
                tmp = t[i];
                t[i] = t[j];
                t[j] = tmp;
            }
        }
    }
    // print the sorted multiplicity-threshold pairs:
    for (size_t i = 1; i <= n; i++) {
        printf("%ld: multiplicity %d, threshold %d\n", i, m[i], t[i]);
    }
    // next, generate the partitions
    generate_partitions(m, n, visit_partition);
    printf("Invalid: %d\n", invalid);
    printf("Best cost: %d\n", best_cost);
}
