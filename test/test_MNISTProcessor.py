import pytest
from mnistdp.MNISTProcessor import MNISTProcessor
import os
import numpy as np
from .training_image_10 import train_image


class TestMNISTProcessor:
    m = None

    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        cls.m = MNISTProcessor("test/config_default.ini")

    @classmethod
    def teardown_class(cls):
        """ teardown any state that was previously setup with a call to
        setup_class.
        """

    def test_init(self):
        MNISTProcessor("test/config_default.ini")

    def test_init_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            MNISTProcessor("test/config_nonexist.ini")

    def test_train_image_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            MNISTProcessor("test/config_train_image_nonexist.ini")

    def test_train_label_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            MNISTProcessor("test/config_train_label_nonexist.ini")

    def test_test_image_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            MNISTProcessor("test/config_test_image_nonexist.ini")

    def test_test_label_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            MNISTProcessor("test/config_test_label_nonexist.ini")

    def test_train_image_assignment(self):
        assert self.m.tr_images == "/home/tylee/01_dataset/MNIST/train-images-idx3-ubyte"

    def test_train_label_assignment(self):
        assert self.m.tr_labels == "/home/tylee/01_dataset/MNIST/train-labels-idx1-ubyte"

    def test_test_image_assignment(self):
        assert self.m.t10k_images == "/home/tylee/01_dataset/MNIST/t10k-images-idx3-ubyte"

    def test_test_label_assignment(self):
        assert self.m.t10k_labels == "/home/tylee/01_dataset/MNIST/t10k-labels-idx1-ubyte"

    def test_train_generator(self):
        gen = self.m.generator(batch_size=1)
        i = 0
        try:
            while i < 5:
                image, label = gen.__next__()  # type:np.ndarray
                assert np.array_equal(image.squeeze(), train_image[i, :, :].squeeze())
                i += 1
        except StopIteration:
            pass

    @pytest.mark.parametrize("bs", range(1, 128, 4))
    def test_train_generator_length(self, bs):
        gen = self.m.generator(batch_size=bs)
        i = 0
        try:
            while True:
                gen.__next__()
                i += 1
        except StopIteration:
            pass
        assert i == np.ceil(60000 / bs)

    @pytest.mark.parametrize("bs", range(1, 128, 4))
    def test_test_generator_length(self, bs):
        gen = self.m.generator(mode="test", batch_size=bs)
        i = 0
        try:
            while True:
                gen.__next__()
                i += 1
        except StopIteration:
            pass
        assert i == np.ceil(10000 / bs)

    @pytest.mark.parametrize("bs", range(1, 32, 1))
    def test_test_generator_shape(self, bs):
        gen = self.m.generator(mode="test", batch_size=bs)
        image, label = gen.__next__()
        assert image.shape == (bs, 28, 28)
        assert label.shape == (bs,)

    @pytest.mark.parametrize("bs", range(1, 32, 1))
    def test_train_generator_shape(self, bs):
        gen = self.m.generator(mode="test", batch_size=bs)
        image, label = gen.__next__()
        assert image.shape == (bs, 28, 28)
        assert label.shape == (bs,)
