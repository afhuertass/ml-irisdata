from __future__ import print_function
import tensorflow as tf
import argparse

import csv_helpers

dataset = ["./data-set/iris_training.csv"]

file_length = csv_helpers.fileLen(dataset[0])-1

examples , labels = csv_helpers.input_pipeline( file_length  , dataset , 5 )


with tf.Session() as sess:
    tf.initialize_local_variables().run()
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord )

    try:
        while not coord.should_stop():
            example_batch , label_batch = sess.run([ examples , labels])
            print("Batchhhhh:")
            print(example_batch)
            
    except tf.errors.OutOfRangeError:
        print("shit run out of data")
    finally:
        coord.request_stop()
    coord.join(threads)


#print(file_length)




