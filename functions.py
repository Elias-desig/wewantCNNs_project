def layer_dimensions(a, b, steps):
    if steps < 2:
        raise ValueError("num layer must be at least 2")
    if b < 1:
        raise ValueError("Output dimensions must be at least 1")
    total_diff = b - a
    num_deltas = steps - 1
    base_step = total_diff // num_deltas
    remainder = total_diff % num_deltas
    # To handle negative remainders correctly
    if remainder < 0:
        base_step += 1
        remainder -= num_deltas
    # Create deltas
    deltas = [base_step] * num_deltas
    for i in range(abs(remainder)):
        deltas[i] += -1 if remainder < 0 else 1
    # Generate the sequence
    out = [a]
    for d in deltas:
        out.append(out[-1] + d)
    return out
