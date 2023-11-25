import numpy as np

def compare_embeddings(embeddings, index = False, verbose = False):
    """
    Compare all pairs of embeddings in the given list of embeddings.

    Parameters
    ----------
    embeddings : list of numpy.ndarray
        The embeddings to compare.
    index : bool, optional
        Whether to return the indices of the embeddings that are equal.
    verbose : bool, optional
        Whether to print a summary of the number of equal embeddings.

    Returns
    -------
    num_equal : int
        The number of equal embeddings.
    equal_indices : list of tuple of int (optional)
        The indices of the embeddings that are equal.
    """
    num_equal = 0
    equal_indices = []

    # TODO: maybe linear complexity possible?
    for i in range(0, len(embeddings)):
        for j in range(i+1, len(embeddings)):
            if equal_embeddings(embeddings[i], embeddings[j]):
                num_equal += 1
                equal_indices.append((i, j))

    num_combinations = len(embeddings) * (len(embeddings) - 1) // 2
    if verbose:
        print("Number of equal embeddings: {}/{}".format(num_equal, num_combinations))
    
    if index:
        return num_equal, equal_indices
    else:
        return num_equal

def equal_embeddings(embedding1, embedding2):
    """
    Compare two embeddings. The embeddings can be of the following types:
    - numpy.ndarray
    - int
    - str

    Parameters
    ----------
    embedding1 : numpy.ndarray, int, or str
        The first embedding to compare.
    embedding2 : numpy.ndarray, int, or str
        The second embedding to compare.

    Returns
    -------
    equal : bool
        True if the embeddings are equal, False otherwise.
    """
    if type(embedding1) is np.ndarray:
        return np.array_equal(embedding1, embedding2)
    elif type(embedding1) is int:
        return embedding1 == embedding2
    elif type(embedding1) is str:
        return embedding1 == embedding2
    elif type(embedding1) is np.str_: # for numpy arrays of strings
        return embedding1 == embedding2
    else:
        raise TypeError("Embedding type {} not recognized.".format(type(embedding1)))