def round_to_multiple(num: int, multiple: int) -> int:
    """
    A simple utility function to round a number to the nearest multiple.
    This helps us pad sequences to play nice with GPUs.
    """
    return (num // multiple + 1) * multiple
