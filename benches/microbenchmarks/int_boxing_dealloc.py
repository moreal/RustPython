def allocate_uncached_ints(iterations):
    base = 1 << 80
    return [base + index for index in range(iterations)]


values = allocate_uncached_ints(ITERATIONS)

# ---
values = None
