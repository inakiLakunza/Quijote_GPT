



def get_stats(ids, load_counts=None):

    """
    ids is a list of integers (which will be our) input sentence
    converted to bytes and then to integers. 
    And we will return a dictionary with the frequencies of
    each consecutive pair:
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """

    counts = {} if load_counts is None else load_counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):

    """
    EXPLAIN IT
    """
    
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids