import math


def pad_num(num: int, padding_multiple: int = 2):
    digits = 1 if num == 0 else int(math.log10(num)) + 1
    padding_required = ((digits // padding_multiple) + 1) * padding_multiple
    return f"{num:>{padding_required}d}"
