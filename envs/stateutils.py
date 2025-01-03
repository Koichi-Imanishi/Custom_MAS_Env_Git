"""Utility functions to process state."""

import torch


def desired_directions(state):
    """Given the current state and destination, compute desired direction."""
    destination_vectors = state[:, 6:8] - state[:, 0:2]

    # support for prediction without given destination:
    # "desired_direction" is in the direction of the current velocity
    invalid_destination = torch.isnan(destination_vectors[:, 0])
    destination_vectors[invalid_destination] = state[invalid_destination, 2:4]

    norm_factors = torch.linalg.norm(destination_vectors, ord=2, dim=-1)
    norm_factors[norm_factors == 0.0] = 1.0
    return destination_vectors / norm_factors.unsqueeze(-1)


def speeds(state):
    """Return the speeds corresponding to a given state."""
    return torch.linalg.norm(state[:, 2:4], ord=2, dim=-1)

def nearest_point(line_segments, state):
    ps = state[:, 0:2]
    a = line_segments[:, 0:2]
    b = line_segments[:, 2:4]
    ap = ps - a
    ab = b - a
    ba = a - b
    bp = ps - b

    PATTERN_A = (torch.sum(ap * ab, dim=1) < 0).repeat(2, 1).T
    PATTERN_B = ((torch.sum(ap * ab, dim=1) >= 0) & (torch.sum(bp * ba, dim=1) < 0)).repeat(2, 1).T
    PATTERN_C = ((torch.sum(ap * ab, dim=1) >= 0) & (torch.sum(bp * ba, dim=1) >= 0)).repeat(2, 1).T

    ai_norm = torch.sum(ap * ab, dim=1)/torch.norm(ab, dim=1)
    neighbor_point = a * PATTERN_A + b * PATTERN_B + (a + (ab/torch.norm(ab, dim=1).repeat(2, 1).T)*ai_norm.repeat(2, 1).T) * PATTERN_C
    return neighbor_point

def calc_distance(line_segments, state):
    ps = state[:, 0:2]
    a = line_segments[:, 0:2]
    b = line_segments[:, 2:4]
    ap = ps - a
    ab = b - a
    ba = a - b
    bp = ps - b

    PATTERN_A_d = (torch.sum(ap * ab, dim=1) < 0)
    PATTERN_B_d = ((torch.sum(ap * ab, dim=1) >= 0) & (torch.sum(bp * ba, dim=1) < 0))
    PATTERN_C_d = ((torch.sum(ap * ab, dim=1) >= 0) & (torch.sum(bp * ba, dim=1) >= 0))

    ai_norm = torch.sum(ap * ab, dim=1)/torch.norm(ab, dim=1)
    distance = torch.norm(ap, dim=1) * PATTERN_A_d + torch.norm(bp, dim=1) * PATTERN_B_d + torch.norm(ps - (a + (ab/torch.norm(ab, dim=1).repeat(2, 1).T)*ai_norm.repeat(2, 1).T), dim=1) * PATTERN_C_d
    return distance

def calc_distance_1(line_segments, state):
    ps = state[0:2]
    a = line_segments[0:2]
    b = line_segments[2:4]
    ap = ps - a
    ab = b - a
    ba = a - b
    bp = ps - b

    PATTERN_A = (torch.sum(ap * ab) < 0)
    PATTERN_B = ((torch.sum(ap * ab) >= 0) & (torch.sum(bp * ba) < 0))
    PATTERN_C = ((torch.sum(ap * ab) >= 0) & (torch.sum(bp * ba) >= 0))

    ai_norm = torch.sum(ap * ab)/torch.norm(ab)
    distance = torch.norm(ap) * PATTERN_A + torch.norm(bp) * PATTERN_B + torch.norm(ps - (a + (ab/torch.norm(ab).repeat(2, 1).T)*ai_norm.repeat(2, 1).T)) * PATTERN_C
    return distance