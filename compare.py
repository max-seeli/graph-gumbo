from hashlib import sha256

import numpy as np
from tqdm import tqdm


def compare_embeddings(embeddings, index=False, exact=False, verbose=False):
    """
    Compare all pairs of embeddings in the given list of embeddings. The function
    can compare the embeddings exactly or using hashing. If compared exactly, the
    complexity is O(n^2), where n is the number of embeddings. If compared using
    hashing, the average complexity is O(n) with an highly unlikely worst case of 
    O(n^2) if all embeddings hash to the same value.

    Parameters
    ----------
    embeddings : list of numpy.ndarray
        The embeddings to compare.
    index : bool, optional
        Whether to return the indices of the embeddings that are equal.
    exact : bool, optional
        Whether to compare the embeddings exactly. If False, the embeddings are
        compared using hashing.
    verbose : bool, optional
        Whether to print a summary of the number of equal embeddings.

    Returns
    -------
    num_equal : int
        The number of equal embeddings.
    equal_indices : list of tuple of int (optional)
        The indices of the embeddings that are equal.
    """
    if exact:
        num_equal, equal_indices = _compare_embeddings_exact(
            embeddings, verbose)
    else:
        num_equal, equal_indices = _compare_embeddings_hash(
            embeddings, verbose)

    num_combinations = len(embeddings) * (len(embeddings) - 1) // 2
    if verbose:
        print("Number of equal embeddings: {}/{}".format(num_equal, num_combinations))

    if index:
        return num_equal, equal_indices
    else:
        return num_equal


def _compare_embeddings_hash(embeddings, verbose):
    """
    Compare all pairs of embeddings in the given list of embeddings using hashing.

    Parameters
    ----------
    embeddings : list of numpy.ndarray
        The embeddings to compare.

    Returns
    -------
    num_equal : int
        The number of equal embeddings.
    equal_indices : list of tuple of int (optional)
        The indices of the embeddings that are equal.
    """
    num_equal = 0
    equal_indices = set()

    hash_table = {}
    emb_iter = enumerate(embeddings)
    emb_iter = tqdm(emb_iter) if verbose else emb_iter
    for i, embedding in emb_iter:
        hash_value = hash_embedding(embedding)
        if hash_value not in hash_table:
            hash_table[hash_value] = [i]
        else:
            hash_table[hash_value].append(i)

    for hash_value, indices in hash_table.items():
        if len(indices) > 1:
            num_equal += len(indices) * (len(indices) - 1) // 2
            for i in range(0, len(indices)):
                for j in range(i+1, len(indices)):
                    equal_indices.add((indices[i], indices[j]))

    return num_equal, equal_indices


def _compare_embeddings_exact(embeddings, verbose):
    """
    Compare all pairs of embeddings in the given list of embeddings exactly.

    Parameters
    ----------
    embeddings : list of numpy.ndarray
        The embeddings to compare.

    Returns
    -------
    num_equal : int
        The number of equal embeddings.
    equal_indices : list of tuple of int
        The indices of the embeddings that are equal.
    """

    num_equal = 0
    equal_indices = set()

    emb_iter = range(0, len(embeddings))
    emb_iter = tqdm(emb_iter) if verbose else emb_iter
    for i in emb_iter:
        for j in range(i+1, len(embeddings)):
            if equal_embeddings(embeddings[i], embeddings[j]):
                num_equal += 1
                equal_indices.add((i, j))

    return num_equal, equal_indices


def hash_embedding(embedding):
    """
    Hash an embedding.

    Parameters
    ----------
    embedding : numpy.ndarray, int, or str, or set
        The embedding to hash.

    Returns
    -------
    hash_value : str
        The hash value of the embedding.
    """
    t = type(embedding)
    if type(embedding) is np.ndarray:
        return sha256(embedding.tobytes()).hexdigest()
    elif type(embedding) is int:
        return sha256(str(embedding).encode()).hexdigest()
    elif type(embedding) is str:
        return sha256(embedding.encode()).hexdigest()
    elif type(embedding) is np.str_:  # for numpy arrays of strings
        return sha256(embedding.encode()).hexdigest()
    elif type(embedding) is set:
        return sha256(str(sorted(embedding)).encode()).hexdigest()
    else:
        raise TypeError("Embedding type {} not recognized.".format(t))


def equal_embeddings(embedding1, embedding2):
    """
    Compare two embeddings. The embeddings can be of the following types:
    - numpy.ndarray
    - int
    - str
    - set

    Parameters
    ----------
    embedding1 : numpy.ndarray, int, or str, or set
        The first embedding to compare.
    embedding2 : numpy.ndarray, int, or str, or set
        The second embedding to compare.

    Returns
    -------
    equal : bool
        True if the embeddings are equal, False otherwise.
    """
    if type(embedding1) is np.ndarray:
        return np.allclose(embedding1, embedding2)
    elif type(embedding1) is int:
        return embedding1 == embedding2
    elif type(embedding1) is str:
        return embedding1 == embedding2
    elif type(embedding1) is np.str_:  # for numpy arrays of strings
        return embedding1 == embedding2
    elif type(embedding1) is set:
        return embedding1 == embedding2
    else:
        raise TypeError(
            "Embedding type {} not recognized.".format(type(embedding1)))
