"""Formatting utilities used by module representations."""


def _addindent(s_, numSpaces):
    """Indent the body of a multi-line string by a fixed number of spaces."""
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    # Only add indentation to non-blank lines; blank lines stay empty
    s = [(numSpaces * " ") + line if line.strip() else "" for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s
