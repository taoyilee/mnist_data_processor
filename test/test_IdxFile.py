import pytest
from mnistdp import IdxFile
import os


class TestIDXFile:
    idx_tr_images = None
    idx_tr_labels = None
    idx_t10k_images = None
    idx_t10k_labels = None

    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        tr_images = "/home/tylee/01_dataset/MNIST/train-images-idx3-ubyte"
        tr_labels = "/home/tylee/01_dataset/MNIST/train-labels-idx1-ubyte"
        t10k_images = "/home/tylee/01_dataset/MNIST/t10k-images-idx3-ubyte"
        t10k_labels = "/home/tylee/01_dataset/MNIST/t10k-labels-idx1-ubyte"

        cls.idx_tr_images = IdxFile.from_file(tr_images)
        cls.idx_tr_labels = IdxFile.from_file(tr_labels)
        cls.idx_t10k_images = IdxFile.from_file(t10k_images)
        cls.idx_t10k_labels = IdxFile.from_file(t10k_labels)

    @classmethod
    def teardown_class(cls):
        """ teardown any state that was previously setup with a call to
        setup_class.
        """

    def test_from_file(self):
        tr_images = "/home/tylee/01_dataset/MNIST/train-images-idx3-ubyte"
        tr_labels = "/home/tylee/01_dataset/MNIST/train-labels-idx1-ubyte"
        t10k_images = "/home/tylee/01_dataset/MNIST/t10k-images-idx3-ubyte"
        t10k_labels = "/home/tylee/01_dataset/MNIST/t10k-labels-idx1-ubyte"
        idx_file = IdxFile.from_file(tr_images)
        assert isinstance(idx_file, IdxFile)
        idx_file = IdxFile.from_file(tr_labels)
        assert isinstance(idx_file, IdxFile)
        idx_file = IdxFile.from_file(t10k_images)
        assert isinstance(idx_file, IdxFile)
        idx_file = IdxFile.from_file(t10k_labels)
        assert isinstance(idx_file, IdxFile)

    def test_from_file_nonexist(self):
        nonexist_file = "/home/tylee/01_dataset/MNIST/train-images-idx3-ubyte_nonexist"
        with pytest.raises(FileNotFoundError):
            IdxFile.from_file(nonexist_file)

    def test_dimension(self):
        assert self.idx_tr_images.dimension == (60000, 28, 28)
        assert self.idx_tr_labels.dimension == (60000,)
        assert self.idx_t10k_images.dimension == (10000, 28, 28)
        assert self.idx_t10k_labels.dimension == (10000,)

    def test_type(self):
        assert self.idx_tr_images.type_flag == b'\x08'
        assert self.idx_tr_labels.type_flag == b'\x08'
        assert self.idx_t10k_images.type_flag == b'\x08'
        assert self.idx_t10k_labels.type_flag == b'\x08'
