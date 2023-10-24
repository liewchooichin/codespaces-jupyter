import numpy as np
import tensorflow as tf

#test_logdir = get_run_logdir()
test_logdir = "my_logdir/my_logdir1"
writer = tf.summary.create_file_writer(test_logdir)
with writer.as_default():
    for step in range(1, 10 + 1):
        tf.summary.scalar("my_scalar", np.sin(step / 10), step=step)
