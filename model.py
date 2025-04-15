import math
#import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# This function creates an autoencoder model using TensorFlow.
def ae(input_size, code_size,
       corruption=0.0, tight=False,
       enc=tf.nn.tanh, dec=tf.nn.tanh):


    x = tf.placeholder(tf.float32, [None, input_size])

    if corruption > 0.0:
        # Corrupt input data
        _x = tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                                      minval=0,
                                                      maxval=1 - corruption,
                                                      dtype=tf.float32), tf.float32))

    else:
        _x = x

    b_enc = tf.Variable(tf.zeros([code_size]))

    W_enc = tf.Variable(tf.random_uniform(
                [input_size, code_size],
                -6.0 / math.sqrt(input_size + code_size),
                6.0 / math.sqrt(input_size + code_size))
            )

    encode = tf.matmul(_x, W_enc) + b_enc
    if enc is not None:
        encode = enc(encode)

    b_dec = tf.Variable(tf.zeros([input_size]))
    if tight:
        W_dec = tf.transpose(W_enc)

    else:

        # Initialize decoder weights using Glorot method
        W_dec = tf.Variable(tf.random_uniform(
                    [code_size, input_size],
                    -6.0 / math.sqrt(code_size + input_size),
                    6.0 / math.sqrt(code_size + input_size))
                )

    # Compute activation for decoding
    decode = tf.matmul(encode, W_dec) + b_dec
    if dec is not None:
        decode = enc(decode)

    model = {

        "input": x,
        "encode": encode,
        "decode": decode,
        "cost": tf.sqrt(tf.reduce_mean(tf.square(x - decode))),

        "params": {
            "W_enc": W_enc,
            "b_enc": b_enc,
            "b_dec": b_dec,
        }

    }

   
    if not tight:
        model["params"]["W_dec"] = W_dec

    return model

#neural network model
def nn(input_size, n_classes, layers, init=None):
   

    tf.compat.v1.disable_eager_execution()
    input = x = tf.placeholder(tf.float32, [None, input_size])

    y = tf.placeholder("float", [None, n_classes])

    actvs = []
    dropouts = []
    params = {}
    for i, layer in enumerate(layers):
        #print("layer",i)
        
        dropout = tf.placeholder(tf.float32)

        if init is None:
            W = tf.Variable(tf.zeros([input_size, layer["size"]]))
            b = tf.Variable(tf.zeros([layer["size"]]))
            
        else:

            W = tf.Variable(init[i]["W"])
            b = tf.Variable(init[i]["b"])

        x = tf.matmul(x, W) + b

        if "actv" in layer and layer["actv"] is not None:
            x = layer["actv"](x)

        #print(x.tolist())
        x = tf.nn.dropout(x, dropout)
        params.update({
            "W_" + str(i+1): W,
            "b_" + str(i+1): b,
        })
        actvs.append(x)
        dropouts.append(dropout)

        input_size = layer["size"]
      
    # Initialize output layer weights & bias using Glorot method
    W = tf.Variable(tf.random_uniform(
            [input_size, n_classes],
            -3.0 / math.sqrt(input_size + n_classes),
            3.0 / math.sqrt(input_size + n_classes)))
    b = tf.Variable(tf.zeros([n_classes]))

    y_hat = tf.matmul(x, W) + b

    # Add layer parameters
    params.update({"W_out": W, "b_out": b})
    actvs.append(y_hat)

    return {

        "input": input,
        "expected": y,
        "output": tf.nn.softmax(y_hat),
        "cost": tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y)),
        "dropouts": dropouts,
        "actvs": actvs,
        "params": params,
    }
