
�root"_tf_keras_network*�{"name": "model_36", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_36", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_46"}, "name": "input_46", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding_45", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 12528, "output_dim": 128, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": true, "input_length": null}, "name": "embedding_45", "inbound_nodes": [[["input_46", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_45", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_45", "inbound_nodes": [[["embedding_45", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_158", "trainable": true, "dtype": "float32", "units": 101, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_158", "inbound_nodes": [[["lstm_45", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_60", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_60", "inbound_nodes": [[["dense_158", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.transpose_29", "trainable": true, "dtype": "float32", "function": "compat.v1.transpose"}, "name": "tf.compat.v1.transpose_29", "inbound_nodes": [["dropout_60", 0, 0, {"perm": [0, 2, 1], "name": "transpose", "conjugate": false}]]}, {"class_name": "Dense", "config": {"name": "dense_160", "trainable": true, "dtype": "float32", "units": 8080, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_160", "inbound_nodes": [[["tf.compat.v1.transpose_29", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_16", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_16", "inbound_nodes": [["dense_160", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_159", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_159", "inbound_nodes": [[["tf.compat.v1.transpose_29", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_161", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_161", "inbound_nodes": [[["tf.nn.relu_16", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.softmax_28", "trainable": true, "dtype": "float32", "function": "nn.softmax"}, "name": "tf.nn.softmax_28", "inbound_nodes": [["dense_159", 0, 0, {"axis": 1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["dense_161", 0, 0, {"y": ["tf.nn.softmax_28", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.tanh_2", "trainable": true, "dtype": "float32", "function": "math.tanh"}, "name": "tf.math.tanh_2", "inbound_nodes": [["tf.__operators__.add", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_162", "trainable": true, "dtype": "float32", "units": 160, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_162", "inbound_nodes": [[["tf.math.tanh_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_163", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_163", "inbound_nodes": [[["dense_162", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_22", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_22", "inbound_nodes": [["dense_163", 0, 0, {"y": 0.20000000298023224}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.add_22", "trainable": true, "dtype": "float32", "function": "math.add"}, "name": "tf.math.add_22", "inbound_nodes": [["tf.math.multiply_22", 0, 0, {"y": 0.5}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_22", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_22", "inbound_nodes": [["tf.math.add_22", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1.0}]]}], "input_layers": [["input_46", 0, 0]], "output_layers": [["tf.clip_by_value_22", 0, 0]]}, "shared_object_id": 35, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 18]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 18]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 18]}, "float32", "input_46"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 18]}, "float32", "input_46"]}, "keras_version": "2.6.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_36", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_46"}, "name": "input_46", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Embedding", "config": {"name": "embedding_45", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 12528, "output_dim": 128, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 1}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": true, "input_length": null}, "name": "embedding_45", "inbound_nodes": [[["input_46", 0, 0, {}]]], "shared_object_id": 2}, {"class_name": "LSTM", "config": {"name": "lstm_45", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_45", "inbound_nodes": [[["embedding_45", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_158", "trainable": true, "dtype": "float32", "units": 101, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_158", "inbound_nodes": [[["lstm_45", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dropout", "config": {"name": "dropout_60", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_60", "inbound_nodes": [[["dense_158", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.transpose_29", "trainable": true, "dtype": "float32", "function": "compat.v1.transpose"}, "name": "tf.compat.v1.transpose_29", "inbound_nodes": [["dropout_60", 0, 0, {"perm": [0, 2, 1], "name": "transpose", "conjugate": false}]], "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "dense_160", "trainable": true, "dtype": "float32", "units": 8080, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_160", "inbound_nodes": [[["tf.compat.v1.transpose_29", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_16", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_16", "inbound_nodes": [["dense_160", 0, 0, {"name": null}]], "shared_object_id": 16}, {"class_name": "Dense", "config": {"name": "dense_159", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_159", "inbound_nodes": [[["tf.compat.v1.transpose_29", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "Dense", "config": {"name": "dense_161", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_161", "inbound_nodes": [[["tf.nn.relu_16", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.softmax_28", "trainable": true, "dtype": "float32", "function": "nn.softmax"}, "name": "tf.nn.softmax_28", "inbound_nodes": [["dense_159", 0, 0, {"axis": 1}]], "shared_object_id": 23}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["dense_161", 0, 0, {"y": ["tf.nn.softmax_28", 0, 0], "name": null}]], "shared_object_id": 24}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.tanh_2", "trainable": true, "dtype": "float32", "function": "math.tanh"}, "name": "tf.math.tanh_2", "inbound_nodes": [["tf.__operators__.add", 0, 0, {"name": null}]], "shared_object_id": 25}, {"class_name": "Dense", "config": {"name": "dense_162", "trainable": true, "dtype": "float32", "units": 160, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_162", "inbound_nodes": [[["tf.math.tanh_2", 0, 0, {}]]], "shared_object_id": 28}, {"class_name": "Dense", "config": {"name": "dense_163", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_163", "inbound_nodes": [[["dense_162", 0, 0, {}]]], "shared_object_id": 31}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_22", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_22", "inbound_nodes": [["dense_163", 0, 0, {"y": 0.20000000298023224}]], "shared_object_id": 32}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.add_22", "trainable": true, "dtype": "float32", "function": "math.add"}, "name": "tf.math.add_22", "inbound_nodes": [["tf.math.multiply_22", 0, 0, {"y": 0.5}]], "shared_object_id": 33}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_22", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_22", "inbound_nodes": [["tf.math.add_22", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1.0}]], "shared_object_id": 34}], "input_layers": [["input_46", 0, 0]], "output_layers": [["tf.clip_by_value_22", 0, 0]]}}, "training_config": {"loss": {"class_name": "MeanAbsoluteError", "config": {"reduction": "auto", "name": "mean_absolute_error"}, "shared_object_id": 37}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 38}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999974752427e-07, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_46", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_46"}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "embedding_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding_45", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 12528, "output_dim": 128, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 1}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": true, "input_length": null}, "inbound_nodes": [[["input_46", 0, 0, {}]]], "shared_object_id": 2, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18]}}2
�root.layer_with_weights-1"_tf_keras_rnn_layer*�{"name": "lstm_45", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LSTM", "config": {"name": "lstm_45", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.0, "implementation": 2}, "inbound_nodes": [[["embedding_45", 0, 0, {}]]], "shared_object_id": 7, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 39}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 18, 128]}}2
�root.layer_with_weights-2"_tf_keras_layer*�{"name": "dense_158", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_158", "trainable": true, "dtype": "float32", "units": 101, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["lstm_45", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18, 128]}}2
�root.layer-4"_tf_keras_layer*�{"name": "dropout_60", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_60", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_158", 0, 0, {}]]], "shared_object_id": 11}2
�root.layer-5"_tf_keras_layer*�{"name": "tf.compat.v1.transpose_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.transpose_29", "trainable": true, "dtype": "float32", "function": "compat.v1.transpose"}, "inbound_nodes": [["dropout_60", 0, 0, {"perm": [0, 2, 1], "name": "transpose", "conjugate": false}]], "shared_object_id": 12}2
�root.layer_with_weights-3"_tf_keras_layer*�{"name": "dense_160", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_160", "trainable": true, "dtype": "float32", "units": 8080, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tf.compat.v1.transpose_29", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 18}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 101, 18]}}2
�root.layer-7"_tf_keras_layer*�{"name": "tf.nn.relu_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_16", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "inbound_nodes": [["dense_160", 0, 0, {"name": null}]], "shared_object_id": 16}2
�	root.layer_with_weights-4"_tf_keras_layer*�{"name": "dense_159", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_159", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tf.compat.v1.transpose_29", 0, 0, {}]]], "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 18}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 101, 18]}}2
�
root.layer_with_weights-5"_tf_keras_layer*�{"name": "dense_161", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_161", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tf.nn.relu_16", 0, 0, {}]]], "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8080}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 101, 8080]}}2
�
�
�
�root.layer_with_weights-6"_tf_keras_layer*�{"name": "dense_162", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_162", "trainable": true, "dtype": "float32", "units": 160, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tf.math.tanh_2", 0, 0, {}]]], "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 101, 80]}}2
�root.layer_with_weights-7"_tf_keras_layer*�{"name": "dense_163", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_163", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_162", 0, 0, {}]]], "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 160}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 101, 160]}}2
�
�
�
�root.layer_with_weights-1.cell"_tf_keras_layer*�{"name": "lstm_cell_45", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LSTMCell", "config": {"name": "lstm_cell_45", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 6}2
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 46}2
��root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 38}2