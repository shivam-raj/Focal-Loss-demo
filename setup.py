from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow as tf
import tensorflow_datasets as tfds
from data import get_data
from model import get_model


keras = tf.keras


def get_batch(size,BATCH_SIZE=20,SHUFFLE_BUFFER_SIZE = 1000):


    train,val,test,labels=get_data(size,'rock_paper_scissors',3)


    train_batches=train.shuffle(SHUFFLE_BUFFER_SIZE).repeat().batch(BATCH_SIZE)
    val_batches=val.shuffle(SHUFFLE_BUFFER_SIZE).repeat().batch(BATCH_SIZE)
    test_batches=test.batch(BATCH_SIZE).repeat()

    if SHUFFLE_BUFFER_SIZE % BATCH_SIZE != 0:
        parallel_steps = SHUFFLE_BUFFER_SIZE // BATCH_SIZE + 1
    else:
        parallel_steps = SHUFFLE_BUFFER_SIZE // BATCH_SIZE

    return train_batches,val_batches,test_batches,parallel_steps



