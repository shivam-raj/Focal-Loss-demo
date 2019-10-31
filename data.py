from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
keras = tf.keras
def get_data(IMG_SIZE,name,classes):
    wts=(8,1,1)
    splits=tfds.Split.TRAIN.subsplit(weighted=wts)
    (raw_train,raw_val,raw_test),metadata=tfds.load(name,split=list(splits),with_info=True,as_supervised=True)

    def format_example(image, label):
      image = tf.cast(image, tf.float32)
      image = (image/127.5) - 1
      image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
      label=tf.one_hot(label,classes)
      return image, label

    train=raw_train.map(format_example)
    val=raw_val.map(format_example)
    test=raw_test.map(format_example)
    get_label_name = metadata.features['label'].int2str
    return train,val,test,get_label_name