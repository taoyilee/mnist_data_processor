import configparser as cp
import os
from .IdxFile import IdxFile


class MNISTProcessor:


    _tr_images: str = ""
    _tr_labels: str = ""
    _t10k_images: str = ""
    _t10k_labels: str = ""

    @property
    def tr_labels(self):
        return self._tr_labels

    @tr_labels.setter
    def tr_labels(self, value):
        if not os.path.isfile(value):
            raise FileNotFoundError(f"training label file {value} is nonexistent")
        self._tr_labels = value

    @property
    def tr_images(self):
        return self._tr_images

    @tr_images.setter
    def tr_images(self, value):
        if not os.path.isfile(value):
            raise FileNotFoundError(f"training image file {value} is nonexistent")
        self._tr_images = value

    @property
    def t10k_labels(self):
        return self._t10k_labels

    @t10k_labels.setter
    def t10k_labels(self, value):
        if not os.path.isfile(value):
            raise FileNotFoundError(f"test label file {value} is nonexistent")
        self._t10k_labels = value

    @property
    def t10k_images(self):
        return self._t10k_images

    @t10k_images.setter
    def t10k_images(self, value):
        if not os.path.isfile(value):
            raise FileNotFoundError(f"test label file {value} is nonexistent")
        self._t10k_images = value

    def __init__(self, config_file):
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"config file {config_file} is nonexistent")
        self.config_file = config_file
        self.cp = cp.ConfigParser()
        self.cp.read(config_file)

        data_dir = self.cp["DEFAULT"]['data_dir']
        self.tr_images = os.path.join(data_dir, self.cp["DEFAULT"]['train-images-idx3-ubyte'])
        self.tr_labels = os.path.join(data_dir, self.cp["DEFAULT"]['train-labels-idx1-ubyte'])
        self.t10k_images = os.path.join(data_dir, self.cp["DEFAULT"]['t10k-images-idx3-ubyte'])
        self.t10k_labels = os.path.join(data_dir, self.cp["DEFAULT"]['t10k-labels-idx1-ubyte'])
        tr_image_file = IdxFile.from_file(self.tr_images)
