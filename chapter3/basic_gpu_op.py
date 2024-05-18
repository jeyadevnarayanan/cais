import tensorflow as tf
my_list = []
## Iterate through the available GPUs
for device in ["/gpu:0", "/gpu:1"]:
    ## Utilize the TensorFlow device manager
    with tf.device(device):
        x = tf.constant([1,2,3], shape=[1,3])
        y = tf.constant([1,2,3],shape [3,1])
        my_list.append(tf.matmul(x, y))
    with tf.device("/cpu:0"):
        sum_operation = tf.add(x,y)
## Run everything through a session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(sum_operation)