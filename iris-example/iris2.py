
import tensorflow as tf

filename_queue = tf.train.string_input_producer(["./data-set/iris_training.csv"] )

reader = tf.TextLineReader()

key, value = reader.read(filename_queue)

record_defaults = [  [0.0], [0.0], [0.0], [0.0] ,[0.0] ]

col1 , col2, col3, col4, col5 = tf.decode_csv(  value, record_defaults=record_defaults  )




features = tf.pack( [col1 , col2, col3 , col4 ] )


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(15):
        example , label = sess.run([ features , col5])
        print(example)

    coord.request_stop()
    coord.join(threads)

    

