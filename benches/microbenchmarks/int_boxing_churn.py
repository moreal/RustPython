def churn_uncached_ints(iterations):
    base = 1 << 80
    value = base
    for index in range(iterations * 50):
        value = base + index
    return value


# ---
result = churn_uncached_ints(ITERATIONS)
