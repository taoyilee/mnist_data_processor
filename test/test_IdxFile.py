import pytest
from mnistdp import IdxFile
import os
import numpy as np
from .training_image_10 import train_image


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

    def test_length_train_image(self):
        assert len(self.idx_tr_images) == 60000

    def test_length_train_label(self):
        assert len(self.idx_tr_labels) == 60000

    def test_length_test_image(self):
        assert len(self.idx_t10k_images) == 10000

    def test_length_test_label(self):
        assert len(self.idx_t10k_labels) == 10000

    def test_generator_length_train_label(self):
        """
        This test case will only pass if original idx file (from Yann Lecun's webpage) is provided.
        :return:
        """
        gen = self.idx_tr_labels.generator()
        i = 0
        try:
            while True:
                gen.__next__()  # type:np.ndarray
                i += 1
        except StopIteration:
            pass
        assert i == 60000

    @pytest.mark.parametrize("bs", range(1, 32, 1))
    def test_generator_length_train_label_batch(self, bs):
        """
        This test case will only pass if original idx file (from Yann Lecun's webpage) is provided.
        :return:
        """

        gen = self.idx_tr_labels.generator(batch_size=bs)
        i = 0
        try:
            while True:
                gen.__next__()  # type:np.ndarray
                i += 1
        except StopIteration:
            pass
        assert i == np.ceil(60000 / bs)

    def test_generator_data_train_image(self):
        """
        This test case will only pass if original idx file (from Yann Lecun's webpage) is provided.
        :return:
        """
        gen = self.idx_tr_images.generator()
        test_image = 5
        i = 0
        while i < test_image:
            image = gen.__next__()  # type:np.ndarray
            assertion = np.array_equal(image.squeeze(), train_image[i, :, :].squeeze())
            if not assertion:
                print(image)
            assert assertion
            i += 1

    def test_generator_data_train_image_batch(self):
        """
        This test case will only pass if original idx file (from Yann Lecun's webpage) is provided.
        :return:
        """
        gen = self.idx_tr_images.generator(batch_size=4)
        image = gen.__next__()  # type:np.ndarray
        golden_result = train_image[0:4, :, :]
        assert np.array_equal(image, golden_result)

    @pytest.mark.parametrize("bs", range(1, 32, 1))
    def test_last_batch_size_train(self, bs):
        """
        This test case will only pass if original idx file (from Yann Lecun's webpage) is provided.
        :return:
        """
        gen = self.idx_tr_images.generator(batch_size=bs)
        try:
            while True:
                image = gen.__next__()  # type:np.ndarray
        except StopIteration:
            last_batch_size = 60000 % bs if 60000 % bs != 0 else bs
            assert last_batch_size == image.shape[0]
