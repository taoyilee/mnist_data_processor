import numpy as np
import struct


class IdxFile:
    TYPE_MAPPING = {b"\x08": "unsigned byte", b"\x09": "signed byte", b"\x0B": "short (2 bytes)",
                    b"\x0C": "int (4 bytes)", b"\x0D": "float (4 bytes)", b"\x0E": "double (8 bytes)"}
    BYTE_MAPPING = {b"\x08": 1, b"\x09": 1, b"\x0B": 2, b"\x0C": 4, b"\x0D": 4, b"\x0E": 8}
    NP_TYPE_MAPPING = {b"\x08": np.uint8, b"\x09": np.int8, b"\x0B": np.short, b"\x0C": np.int32, b"\x0D": np.float32,
                       b"\x0E": np.double}

    def __init__(self, file_name, dimension, type_flag):
        self.file_name = file_name
        self.dimension = dimension
        self.type_flag = type_flag

    def __len__(self):
        return self.dimension[0]

    def _read_data_sample(self, data_matrix: np.ndarray, fptr):
        for x in np.nditer(data_matrix, op_flags=['readwrite']):
            data_bytes = fptr.read(self.BYTE_MAPPING[self.type_flag])
            data_unpacked = struct.unpack("B", data_bytes)[0]
            x[...] = data_unpacked

    def batch_generator(self, batch_size=1):
        gen = self.generator()
        stop_iteration = False
        while True:
            if stop_iteration:
                raise StopIteration
            data_array = []
            try:
                for b in range(batch_size):
                    image = gen.__next__()
                    data_array.append(image.copy())
            except StopIteration:
                stop_iteration = True
            if len(data_array) > 0:
                yield np.stack(data_array, axis=0)

    def generator(self):
        with open(self.file_name, "rb") as fptr:
            fptr.seek(2)  # skip initial 0x00 0x00
            fptr.seek(2)  # skip magic_number
            fptr.seek(4 * len(self.dimension))  # skip dimension
            if len(self.dimension) == 1:
                data = np.zeros(1, dtype=self.NP_TYPE_MAPPING[self.type_flag])
            else:
                data = np.zeros(tuple(self.dimension[1:]), dtype=self.NP_TYPE_MAPPING[self.type_flag])
            for i in range(len(self)):
                self._read_data_sample(data, fptr)
                yield data

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
        return cls(file_name, dimension_tuple, type_flag)
