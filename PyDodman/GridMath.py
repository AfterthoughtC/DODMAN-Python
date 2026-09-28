from PyDodman import util
from PyDodman.Vector import Vector2



def cosine_similarity(dir1: Vector2, dir2: Vector2) -> float:
    """Calculates the cosine similarity of two vectors.

    Args:
        dir1 (Vector2): The first direction vector
        dir2 (Vector2): The second direction vector

    Returns:
        float: A number between -1 and 1. 1 means the vectors are similar. 0 means the vectors are parallel. -1 means the vectors are opposite.
    """
    numerator: float = dir1.x*dir2.x + dir1.y*dir2.y
    denominator: float = ((dir1.get_length())*(dir2.get_length()))**0.5
    similarity: float = numerator / denominator
    return similarity


def are_two_vectors_parallel(dir1: Vector2, dir2: Vector2, rel_tol=util.RELATIVE_TOLERANCE, abs_tol=util.EPSILON) -> bool:
    """Checks if the two direction vectors are parallel with each other.

    Args:
        dir1 (Vector2): The first direction vector
        dir2 (Vector2): The second direction vector
        rel_tol (float, optional): Relative tolerance. Defaults to util.RELATIVE_TOLERANCE.
        abs_tol (float, optional): Absolute tolerance. Defaults to util.EPSILON.

    Returns:
        bool: True if the two vectors are (almost) parallel and False otherwise.
    """
    similarity: float = cosine_similarity(dir1,dir2)
    return util.is_almost_equal(abs(similarity),1,abs_tol,rel_tol)


