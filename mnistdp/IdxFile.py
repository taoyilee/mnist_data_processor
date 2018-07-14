import numpy as np
import struct


class IdxFile:
    TYPE_MAPPING = {b"\x08": "unsigned byte", b"\x09": "signed byte", b"\x0B": "short (2 bytes)",
                    b"\x0C": "int (4 bytes)", b"\x0D": "float (4 bytes)", b"\x0E": "double (8 bytes)"}

    def __init__(self, dimension, type_flag):
        self.dimension = dimension
        self.type_flag = type_flag

    @classmethod
    def from_file(cls, file_name):
        """
        THE IDX FILE FORMAT
        the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.

        The basic format is

        magic number
        size in dimension 0
        size in dimension 1
        size in dimension 2
        .....
        size in dimension N
        data

        The magic number is an integer (MSB first). The first 2 bytes are always 0.

        The third byte codes the type of the data:
        0x08: unsigned byte
        0x09: signed byte
        0x0B: short (2 bytes)
        0x0C: int (4 bytes)
        0x0D: float (4 bytes)
        0x0E: double (8 bytes)

        The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....

        The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).

        The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.
        """
        with open(file_name, "rb") as fptr:
            fptr.read(2)  # discard initial 0x00 0x00
            magic_number = struct.unpack("cb", fptr.read(2))
            type_flag = magic_number[0]
            dimension = magic_number[1]
            dimension_list = [0 for _ in range(dimension)]
            for d in range(dimension):
                size_i = struct.unpack(">I", fptr.read(4))  # Dimension is represented in 4 bytes
                dimension_list[d] = size_i[0]
            dimension_tuple = tuple(dimension_list)
        return cls(dimension_tuple, type_flag)
