def allocate_header_heavy_objects(iterations):
    base = 1 << 80
    integers = [base + index for index in range(iterations * 20)]
    tuples = [(value, value + 1) for value in integers]
    return integers, tuples


# ---
result = allocate_header_heavy_objects(ITERATIONS)
