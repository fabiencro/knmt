from nltk import edit_distance


def fuzzy_char_level_similarity(x, y):
    return 1 - edit_distance(x, y) * 1.0 / max(len(x), len(y))


def fuzzy_word_level_similarity(x, y, sep=None):
    xs = x.split(sep)
    ys = y.split(sep)
    return fuzzy_char_level_similarity(xs, ys)
