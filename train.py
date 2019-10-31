from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from model import get_model
from setup import get_batch
from loss.focal_loss import focal_loss

print(tf.__version__)


base_lr=0.0001
initial_epochs = 25
validation_steps = 20
IMG_SIZE=300
IMG_SHAPE=(IMG_SIZE,IMG_SIZE,3)

model=get_model(IMG_SHAPE)
train_batches,val_batches,test_batches,steps_per_epoch=get_batch(IMG_SIZE)


model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_lr),
              loss=[focal_loss(alpha=.25, gamma=2)],  #loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()



loss0,accuracy0 = model.evaluate(val_batches, steps = validation_steps)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    steps_per_epoch = steps_per_epoch,
                    validation_steps=validation_steps,
                    validation_data=val_batches)

