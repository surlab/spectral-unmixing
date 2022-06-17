print(
    "this is a really long formatted string that should violate the style guides I think testing black seems like I need to make it even LONGER!!!!"
)


def very_important_function(
    template: str,
    *variables,
    file: os.PathLike,
    debug: bool = False,
):
    """Applies `variables` to the `template` and writes to `file`."""
    with open(file, "w") as f:
        ...
