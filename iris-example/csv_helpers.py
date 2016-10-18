
from __future__ import print_function

import numpy as np
import tensorflow as tf

# encontrar el numero de lineas a leer en el csv

def fileLen(fileName):
    with open(fileName) as f:
        for i , l in enumerate(f):
            pass
    return i+1

def readCsv(filename_queue):
    reader = tf.TextLineReader( )

    key , value = reader.read(filename_queue)

    record_defaults = [[0.0 ] ,[ 0.0  ], [ 0.0] ,[ 0.0] , [0.0] ]
    col1, col2, col3 , col4 , targets = tf.decode_csv( value , record_defaults=record_defaults)

    features = tf.pack([ col1 , col2, col3 , col4])
    label =  tf.pack([targets])

    return features, label


def input_pipeline(batch_size , num_epochs=None, dataset ):
    # argumento files debe ser una lista con los archivos
    filename_queue = tf.train.string_input_producer( dataset , num_epochs=num_epochs )

    min_after_dequeue = 1000 # ver significado
    capacity = min_after_dequeue + 3*batch_size # checkiar significado
    example , label = readCsv(filename_queue)

    examble_batch , label_batch = tf.train.shuffle_batch( [example,label],
                                                          batch_size=batch_size , capacity = capacity , min_after_dequeue = min_after_dequeue
    )

    return example_batch , label_batch



