import pytest
from app.MNISTProcessor import MNISTProcessor
import os


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
