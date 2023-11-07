# PART 2
def num_of_wc_runs(n, m):
    memo = {}

    def dp(n, m):
        if (n, m) in memo:
            return memo[(n, m)]

        if m == 0:
            return 0
        if n == 1:
            return m - 1

        min_runs = float('inf')
        for k in range(1, n + 1):
            runs = 1 + max(dp(k - 1, m - 1), dp(n - k, m))
            min_runs = min(min_runs, runs)

        memo[(n, m)] = min_runs
        return min_runs

    return dp(n, m)


def next_setting(n, m):
    if m == 1:
        return 1  # Linear search with one brick

    # To minimize the worst-case runs, choose k that splits the remaining settings roughly in half.
    return (n + 1) // 2

