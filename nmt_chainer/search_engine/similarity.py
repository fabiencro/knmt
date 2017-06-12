from Levenshtein import distance


def fuzzy_char_level_similarity(x, y):
    return 1 - distance(x, y) / max(len(x), len(y))
