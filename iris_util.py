import tensorflow as tf
import tensorflow_transform as tft

from typing import List
from absl import logging

from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils

from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2

FEATURE_KEYS = [
   "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"
]
LABEL_KEY = "Species"

_TRAIN_BATCH_SIZE = 32 # TODO check
_EVAL_BATCH_SIZE = 16 # TODO check

# FEATURE_SPEC = {
#     **{
#         feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for feature in FEATURE_KEYS
#     },
#     LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
# }

def preprocessing_fn(inputs):
    """Callback function for preprocessing inputs
    Args:
        inputs: map from feature keys to raw features
    Returns:
        map from string feature key to transformed feature.
    """
    outputs = {}
    for feature in FEATURE_KEYS:
        outputs[feature] = tft.scale_to_z_score(inputs[feature])
    outputs[LABEL_KEY] = tft.compute_and_apply_vocabulary(
        inputs[LABEL_KEY],
        default_value=-1,
        vocab_filename="iris_vocab",
        frequency_threshold=3
    )
    return outputs

def _apply_preprocessing(raw_features, tft_layer):
    """
        Apply preprocessing to training data and serving requests

        Args:
            raw_features: raw features or requests
            tft_layer: Tensorflow layer for transforming inputs
    """
    transformed_features = tft_layer(raw_features)
    if LABEL_KEY in raw_features:
        transformed_label = transformed_features.pop(LABEL_KEY)
        return transformed_features, transformed_label
    else:
        return transformed_features, None

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """
        Handler function that gets a serialized tf.example, preprocess and predict
        Args:
            model: current model
            tf_transform_output: transform graph from Transformer
    """

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")])
    def serve_tf_examples_fn(serialized_tf_examples):
        """
            Serialize input into tf.Examples and perform transformation
            Args
                serialized_tf_examples: String
        """
        feature_spec = tf_transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        # Preprocessing
        transformed_features = _apply_preprocessing(parsed_features, model.tft_layer)
        # Predict
        return model(transformed_features)

    return serve_tf_examples_fn

def _input_fn(
        file_pattern: List[str],
        data_accessor: tfx.components.DataAccessor,  # Input to record batch
        tft_transform_output: tft.TFTransformOutput,
        batch_size: int = 50
) -> tf.data.Dataset:  # (features: dict of tensors, indices: label indices)
    """
        Generates features and label for tuning/training
        Args:
            file_pattern: List of paths or patterns of input tfrecord files
            data_accessor: Data Accessor for converting input to RecordBatch.
            tf_transform_output: A TFTransformOuptut.
            batch_size: size of batch
        Returns:
            Dataset containes features and labels
    """
    ds_options = tfxio.TensorFlowDatasetOptions(
        batch_size=batch_size
    )

    dataset = data_accessor.tf_dataset_factory(
        file_pattern, ds_options, schema=tft_transform_output.raw_metadata.schema)

    transform_layer = tft_transform_output.transform_features_layer()

    # Apply preprocessing for each data in dataset
    return dataset.map(lambda x: _apply_preprocessing(x, transform_layer)).repeat()

def _build_model():
    inputs = [keras.layers.Input(shape=(1,), name=f) for f in FEATURE_KEYS]
    input_layer = keras.layers.concatenate(inputs)            
    d1 = keras.layers.Dense(20, kernel_initializer="glorot_uniform", activation="selu")(input_layer)
    d2 = keras.layers.Dense(10, kernel_initializer="glorot_uniform", activation="selu")(d1)
    output_layer = keras.layers.Dense(3, activation="softmax")(d2)
    model = keras.Model(inputs=inputs, outputs=output_layer)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=[keras.metrics.SparseCategoricalAccuracy()])
    model.summary(print_fn=logging.info)
    return model

def run_fn(fn_args: tfx.components.FnArgs):
    tft_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # schema = schema_utils.schema_from_feature_spec(FEATURE_SPEC)
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tft_transform_output,
        _TRAIN_BATCH_SIZE
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tft_transform_output,
        _EVAL_BATCH_SIZE
    )

    model = _build_model()

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps
    )

    model.save(fn_args.serving_model_dir, save_format="tf")