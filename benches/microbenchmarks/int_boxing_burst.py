def allocate_uncached_ints(iterations):
    base = 1 << 80
    return [base + index for index in range(iterations)]


# ---
result = allocate_uncached_ints(ITERATIONS)
