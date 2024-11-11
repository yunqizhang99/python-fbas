#include <stdio.h>
#include <stdlib.h> // Include for atoi
#include <limits.h> // Include for INT_MAX
#include <string.h>
#include "multi-partitions.h"

#define max(a, b) ((a) < (b) ? (b) : (a))

int *t; // array of thresholds, which must be populated before calling cost
int hyper_edges_only = 0;

// Given a partition, compute its cost (number of edges)
// If the thresholds are not satisfied, cost is INT_MAX
int cost(int c[], int v[], int f[], int l) {
    // first compute the cardinality of each part:
    int card[l+1];
    for (int k = 0; k < l+1; k++) {
        card[k] = 0;
        for (int i = f[k]; i < f[k+1]; i++)
            card[k] += v[i];
    }
    // then, check whether all the degree constraints are satisfied
    // this assumes that the components are sorted in decreasing threshold order (i.e. t[1] >= t[2] >= ...)
    for (int k = 0; k < l+1; k++) {
        // compute number of hyper-edges of each organization in the part k
        // we start with the intra-part edges
        int e[card[k]];
        // first count the intra-part edges:
        for (int i = 0; i < card[k]; i++)
            e[i] = card[k]; // not -1 because "self" counts as an edge in this case
        // next count the inter-part edges:
        for (int j = 0; j < l+1; j++) {
            if (j == k) continue;
            if (card[j] < card[k]) {
                for (int i = 0; i < card[k]; i++)
                    e[i] += 1;
            }
            else {
                // the first card[j] % card[k] organizations get one more edge
                for (int i = 0; i < card[j] % card[k]; i++)
                    e[i] += (card[j] / card[k]) + 1;
                // the rest get card[j] / card[k] edges
                for (int i = card[j] % card[k]; i < card[k]; i++)
                    e[i] += card[j] / card[k];
            }
        }
        // finally, check that each organization has at least its threshold number of edges
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
    // next, if we have returned INT_MAX, compute the cost of the partition
    int sum = 0;
    // then, sum all intra-part hyper-edges:
    for (int k = 0; k < l+1; k++)
        if (hyper_edges_only)
            sum += (card[k] * (card[k] - 1)) / 2;
        else
            // 6 * (the edge cardinality of the complete graph with card vertices), plus 3*card:
            sum += 3 * card[k] * (card[k] - 1) + 3 * card[k];
    // finally, iterate over all unordered pairs of parts and sum the inter-part hyper-edges:
    for (int i = 0; i < l; i++)
        for (int j = i+1; j < l+1; j++)
            if (hyper_edges_only)
                sum += max(card[i], card[j]);
            else
                sum += 9*max(card[i], card[j]);
    return sum;
}

int best_cost = INT_MAX;
// int best_cost_all = INT_MAX;
int invalid = 0;
void visit_partition(int c[], int v[], int f[], int l) {
    int cst = cost(c, v, f, l);
    if (cst == INT_MAX)
        invalid++;
    if (cst <= best_cost) {
        best_cost = cst;
        print_partition(c, v, f, l);
        printf("Cost: %d\n\n", cst);
    }
}

int main(int argc, char *argv[]) {
    // first check if --hyper-edges-only is specified
    if (argc > 1 && strcmp(argv[1], "--hyper-edges-only") == 0) {
        hyper_edges_only = 1;
        argc--;
        argv++;
    }
    // the user is expected to provide a list of pairs of numbers whose first element specifies a multiplicity and whose second element specifies a threshold value.
    // for example, 3 2 4 3 specifies that we have 3 organization with threshold 2 and 4 organizations with threshold 3
    if (argc % 2 != 1 || argc < 3) {
        printf("argc: %d\n", argc);
        printf("Usage: %s [--hyper-edges-only] m1 t1 m2 t2 ...\n", argv[0]);
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
    // Now sort the components in decreasing threshold order (a precondition for the cost function)
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
    printf("Best cost appears above\n");
}