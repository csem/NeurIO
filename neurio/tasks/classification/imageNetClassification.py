#!/user/bin/env python

"""
Author: Romain Gaulier
Email: romain.gaulier@csem.ch
Copyright: CSEM, 2023
Creation: 26.04.23
Description: Pipeline for MNIST dataset
"""


import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

import sys
import os
import subprocess
import wget
from pycocotools.cocoeval import COCOeval

from neurio.tasks.task import Task


class ImageNetClassification(Task):
    def __init__(self):
        super().__init__()
        self.x = None
        self.y = None

        project_path = os.getcwd()
        # Download ImageNet dataset
        url = 'http://image-net.org/image/ILSVRC2017/ILSVRC2017_CLS-LOC.tar.gz'
        save_dir = 'datasets'
        save_filename = 'ILSVRC2017_CLS-LOC.tar.gz'
        save_path = os.path.join(project_path, save_dir, save_filename)
        wget.download(url, save_path)

        # Load dataset
        image_size = (224, 224) # example to be discussed
        batch_size = 32 # example to be discussed
        self.dataset = tf.keras.preprocessing.image_dataset_from_directory(
            save_path,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="categorical")

        # Download COCO
        coco_api_dir = os.path.join(project_path, 'cocoApi')
        subprocess.call(["git", "clone", "https://github.com/cocodataset/cocoapi.git", coco_api_dir])

        # Run setup.py script of COCO API
        setup_script = os.path.join(coco_api_dir, "PYTHONAPI", "setup.py")
        subprocess.call(["python", setup_script, "build_ext", "--inplace"])

        # Download COCO annotations
        annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        annotations_dir = os.path.join(project_path, 'annotations')
        subprocess.call(["wget", annotations_url, "-P", annotations_dir])
        annotations_zip = os.path.join(annotations_dir, "annotations_trainval2017.zip")
        subprocess.call(["unzip", annotations_zip, "-d", annotations_dir])

        self.coco_gt = COCO(annotations_dir + 'instances_val2017.json')
        coco_eval = COCOeval(self.coco_gt)
        self.metric = coco_eval

        self.prepare_data()

    def preprocess(self):
        """
        Preprocess the data
        :return:
        """
        (x_train, y_train), (x_test, y_test) = self.dataset.load_data()

        x_train = np.repeat(np.expand_dims(x_train, axis=3), 3, axis=3)
        x_test = np.repeat(np.expand_dims(x_test, axis=3), 3, axis=3)

        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        # Normalize value
        x_train = (x_train / 255.0).astype(np.float32)
        x_test = (x_test / 255.0).astype(np.float32)

        self.x = x_train.numpy()
        # Not sure?
        self.y = y_train.numpy()

    def evaluate(self, y_train, y_pred):
        self.metric.cocoDt = self.coco_gt.loadRes(y_pred)
        self.metric.evaluate()
        self.metric.accumulate()
        self.metric.summarize()
        mAP = self.metric.stats[0]
        return mAP.numpy()
