import os

import pycls.core.logging as logging
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from .base import BaseDataset


logger = logging.get_logger(__name__)


class ImageNet(BaseDataset):

    def __init__(self, data_path, split):
        super(ImageNet, self).__init__(split)
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "val", "test"]
        assert split in splits, "Split '{}' not supported for ImageNet".format(split)
        logger.info("Constructing ImageNet {}...".format(split))
        self.data_path = data_path
        self.split = split
        self._construct_imdb()

    def _construct_imdb(self):
        split_path = os.path.join(self.data_path, self.split)
        logger.info("{} data path: {}".format(self.split, split_path))

        if self.split in ['train', 'val']:
            self.database = ImageFolder(root=split_path)
        else:  # 'test' split
            self._imdb = []
            im_dir = split_path
            for im_name in os.listdir(im_dir):
                im_path = os.path.join(im_dir, im_name)
                self._imdb.append({"im_path": im_path, "class": -1})  # Test images don't have labels

        if self.split != 'test':
            logger.info("Number of images: {}".format(len(self.database.imgs)))
            logger.info("Number of classes: {}".format(len(self.database.classes)))
        else:
            logger.info("Number of test images: {}".format(len(self._imdb)))

    def __len__(self):
        if self.split != 'test':
            return len(self.database)
        return len(self._imdb)

    def _get_data(self, index):
        if self.split != 'test':
            img, label = self.database[index]
        else:
            img = Image.open(self._imdb[index]["im_path"]).convert('RGB')
            label = self._imdb[index]["class"]
        return img, label
