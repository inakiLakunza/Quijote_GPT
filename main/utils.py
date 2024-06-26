



def get_stats(ids: list[int], load_counts: dict[tuple[int, int], int] = None ) -> dict[tuple[int, int], int]:

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


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:

    """
    Find the given pair in the input sequence (ids)
    and replace it with the given idx. We will be replacing 
    the most common pair with a new merge.
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



def count_total_tokens(list_of_chunks):
    total = sum(len(sublist) for sublist in list_of_chunks)
    return total