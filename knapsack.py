from typing import List, Tuple
import time
import matplotlib.pyplot as plt
import random

# Implementation 1: Brute Force
def ks_brute_force(items: List[Tuple[int]], capacity: int) -> int:
    def subset_value(subset):
        total_weight, total_value = 0, 0
        for i in range(len(subset)):
            if subset[i]:
                total_weight += items[i][0]
                total_value += items[i][1]
        return total_value if total_weight <= capacity else 0

    max_value = 0
    n = len(items)
    for i in range(2 ** n):
        subset = [int(x) for x in bin(i)[2:].zfill(n)]
        max_value = max(max_value, subset_value(subset))

    return max_value


# Implementation 2: Recursive
def ks_rec(items: List[Tuple[int]], capacity: int) -> int:
    def knapsack_recursive(i, j):
        if i == 0 or j == 0:
            return 0
        if items[i - 1][0] > j:
            return knapsack_recursive(i - 1, j)
        else:
            return max(
                knapsack_recursive(i - 1, j),
                knapsack_recursive(i - 1, j - items[i - 1][0]) + items[i - 1][1]
            )

    return knapsack_recursive(len(items), capacity)


# Dynamic Programming - Bottom-Up
def ks_bottom_up(items: List[Tuple[int]], capacity: int) -> int:
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, capacity + 1):
            if items[i - 1][0] > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - items[i - 1][0]] + items[i - 1][1])

    return dp[n][capacity]


# Dynamic Programming - Top-Down (Memoization)
def ks_top_down(items: List[Tuple[int]], capacity: int) -> int:
    n = len(items)
    memo = [[-1] * (capacity + 1) for _ in range(n + 1)]

    def knapsack_recursive(i, j):
        if i == 0 or j == 0:
            return 0
        if memo[i][j] != -1:
            return memo[i][j]
        if items[i - 1][0] > j:
            memo[i][j] = knapsack_recursive(i - 1, j)
        else:
            memo[i][j] = max(
                knapsack_recursive(i - 1, j),
                knapsack_recursive(i - 1, j - items[i - 1][0]) + items[i - 1][1]
            )
        return memo[i][j]

    return knapsack_recursive(n, capacity)


def generate_random_items(num_items, min_weight, max_weight, min_value, max_value):
    items = []
    for _ in range(num_items):
        weight = random.randint(min_weight, max_weight)
        value = random.randint(min_value, max_value)
        items.append((weight, value))
    return items


# Brute force experiment
def experiment_rec_vs_brute_force():
    num_items_range = range(0, 20)
    rec_runtimes = []
    brute_force_runtimes = []

    for num_items in num_items_range:
        items = generate_random_items(num_items, 10, 30, 50, 100)

        start_time = time.time()
        ks_rec(items, 200)
        rec_runtime = time.time() - start_time
        rec_runtimes.append(rec_runtime)

        start_time = time.time()
        ks_brute_force(items, 200)
        brute_force_runtime = time.time() - start_time
        brute_force_runtimes.append(brute_force_runtime)

    plt.plot(num_items_range, rec_runtimes, label='Recursive')
    plt.plot(num_items_range, brute_force_runtimes, label='Brute Force')
    plt.xlabel('Number of Items')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
    plt.title('Runtime Comparison: Recursive vs. Brute Force')
    plt.show()


# Experiment 1
def experiment_td_vs_bu_td_better():
    num_items_range = range(0, 100)
    td_runtimes = []
    bu_runtimes = []

    for num_items in num_items_range:
        items = generate_random_items(num_items, 10, 30, 50, 100)

        start_time = time.time()
        ks_top_down(items, 200)
        td_runtime = time.time() - start_time
        td_runtimes.append(td_runtime)

        start_time = time.time()
        ks_bottom_up(items, 200)
        bu_runtime = time.time() - start_time
        bu_runtimes.append(bu_runtime)

    plt.plot(num_items_range, td_runtimes, label='Top-Down')
    plt.plot(num_items_range, bu_runtimes, label='Bottom-Up')
    plt.xlabel('Number of Items')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
    plt.title('Experiment 2: TD vs. BU (BU Better)')
    # plt.title('Experiment 1: TD vs. BU (TD Better)')
    plt.show()


# Experiment 2
def experiment_td_vs_bu_bu_better():
    num_items_range = range(0, 100)
    td_runtimes = []
    bu_runtimes = []

    for num_items in num_items_range:
        items = generate_random_items(num_items, 50, 100, 1000, 2000)

        start_time = time.time()
        ks_top_down(items, 200)
        td_runtime = time.time() - start_time
        td_runtimes.append(td_runtime)

        start_time = time.time()
        ks_bottom_up(items, 200)
        bu_runtime = time.time() - start_time
        bu_runtimes.append(bu_runtime)

    plt.plot(num_items_range, td_runtimes, label='Top-Down')
    plt.plot(num_items_range, bu_runtimes, label='Bottom-Up')
    plt.xlabel('Number of Items')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
    # plt.title('Experiment 2: TD vs. BU (BU Better)')
    plt.title('Experiment 1: TD vs. BU (TD Better)')
    plt.show()

# experiment_rec_vs_brute_force()
# experiment_td_vs_bu_td_better()
# experiment_td_vs_bu_bu_better()
