import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist



def center_distance(a_points, b_points):
    if len(a_points) == 0 or len(b_points) == 0:
        return np.zeros((len(a_points), len(b_points)), dtype=np.float32)

    a_points = np.asarray(a_points, dtype=np.float32).reshape(-1, 2)
    b_points = np.asarray(b_points, dtype=np.float32).reshape(-1, 2)
    return cdist(a_points, b_points, metric="euclidean").astype(np.float32)


def linear_assignment(cost_matrix, thresh):
    cost_matrix = np.asarray(cost_matrix, dtype=np.float32)
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

    finite_mask = np.isfinite(cost_matrix)
    valid_mask = finite_mask & (cost_matrix <= float(thresh))
    if not valid_mask.any():
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

    rows_count, cols_count = cost_matrix.shape
    unmatched_cost = max(float(thresh), 0.0) + 1.0
    large_cost = max(float(np.max(cost_matrix[finite_mask])), unmatched_cost) + 1.0

    assignment_cost = cost_matrix.copy()
    assignment_cost[~valid_mask] = large_cost
    augmented_cost = np.full(
        (rows_count + cols_count, cols_count + rows_count),
        large_cost,
        dtype=np.float32,
    )
    augmented_cost[:rows_count, :cols_count] = assignment_cost
    augmented_cost[:rows_count, cols_count:] = large_cost
    augmented_cost[rows_count:, :cols_count] = large_cost
    augmented_cost[rows_count:, cols_count:] = 0.0

    for row in range(rows_count):
        augmented_cost[row, cols_count + row] = unmatched_cost
    for col in range(cols_count):
        augmented_cost[rows_count + col, col] = unmatched_cost

    rows, cols = linear_sum_assignment(augmented_cost)
    matches, unmatched_a, unmatched_b = [], [], []

    assigned_a = set()
    assigned_b = set()
    for row, col in zip(rows, cols):
        if row >= rows_count or col >= cols_count:
            continue
        if not valid_mask[row, col]:
            continue
        matches.append([row, col])
        assigned_a.add(row)
        assigned_b.add(col)

    for row in range(rows_count):
        if row not in assigned_a:
            unmatched_a.append(row)
    for col in range(cols_count):
        if col not in assigned_b:
            unmatched_b.append(col)

    return np.asarray(matches, dtype=int), tuple(unmatched_a), tuple(unmatched_b)
