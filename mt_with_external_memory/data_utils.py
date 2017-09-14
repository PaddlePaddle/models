"""
    Contains data utilities.
"""


def reader_append_wrapper(reader, append_tuple):
    """
    Data reader wrapper for appending extra data to exisiting reader.
    """

    def new_reader():
        for ins in reader():
            yield ins + append_tuple

    return new_reader
