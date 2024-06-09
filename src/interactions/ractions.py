import jax.numpy as jnp
from typing import Tuple
from .interac_types import InteractionType, InvalidInteractionError, InteractionsInput


def get_interactions(n: int, interac_input: InteractionsInput = InteractionsInput()) -> Tuple:
    """
    Get specified interactions from an n by n lattice using the Euclidean distances.
    Assume a unit distance (1) between nearest neighbours.

    Parameters
    ---
    n: integer representing a side of the square
    interaction_type: InteractionType
    criticality: Criticality

    Output
    ---
    Result containing tuple[unique_pairs, multipliers] or an error
    """

    omega = interac_input.Omega
    rb = interac_input.rydberg_blockade
    interaction_type = interac_input.interaction_type
    numerator = omega * pow(rb, 6)

    # Create a grid of coordinates
    x, y = jnp.meshgrid(jnp.arange(n), jnp.arange(n))
    coordinates = jnp.stack([x.flatten(), y.flatten()], axis=1)

    # Calculate distances between all unique pairs
    num_points = coordinates.shape[0]
    distances = jnp.sqrt(
        jnp.sum((coordinates[:, None, :] - coordinates[None, :, :]) ** 2, axis=-1)
    )

    # Mask to select only unique pairs
    mask = jnp.triu(jnp.ones((num_points, num_points), dtype=bool), k=1)

    if interaction_type == InteractionType.NN:
        mask &= (distances == 1)  # nearest neighbors have unit distance
    elif interaction_type == InteractionType.NNN:
        mask &= (distances == jnp.sqrt(2))  # next nearest neighbors

    # Extract unique pairs, distances, and calculate multipliers
    unique_pairs = jnp.argwhere(mask)
    unique_distances = distances[mask]
    
    denominator = unique_distances ** 6
    multipliers = numerator / denominator

    return unique_pairs, multipliers

