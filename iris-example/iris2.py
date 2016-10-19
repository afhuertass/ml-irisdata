from __future__ import print_function
import tensorflow as tf
import argparse
import numpy as np
import csv_helpers

dataset = ["./data-set/iris_training.csv"]

dataset_test = ["./data-set/iris_test.csv"]

file_length = csv_helpers.fileLen(dataset[0])-1
file_length_test = csv_helpers.fileLen(dataset_test[0]) -1

examples , labels = csv_helpers.input_pipeline( file_length  , dataset , 1 )

test_features , test_labels = csv_helpers.input_pipeline( file_length_test , dataset_test,  1 ) 

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

classifier = tf.contrib.learn.DNNClassifier( feature_columns=feature_columns, hidden_units=[10,20,10] , n_classes=3 , model_dir="./model-iris")

with tf.Session() as sess:
    tf.initialize_local_variables().run()
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord )
    
    try:
        while not coord.should_stop():
            example_batch , label_batch = sess.run([ examples , labels])
            test_batch , label_test_batch = sess.run([ test_features , test_labels ])
            classifier.fit( x = example_batch , y = label_batch , steps=1000)
            
            accuracy_score = classifier.evaluate(x=test_batch ,
                                                 y=label_test_batch )["accuracy"]
            print('Accuracy: {0:f}'.format(accuracy_score))
            
            
            print("Batchhhhh:")
            #print(label_test_batch)
            
    except tf.errors.OutOfRangeError:
        print("shit run out of data")
    finally:
        coord.request_stop()
    coord.join(threads)


print("wtf happen")


new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
y = classifier.predict(new_samples , as_iterable=True)
for i in y:
    print(i)
