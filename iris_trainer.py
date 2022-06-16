from typing import List
from absl import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils

from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2

FEATURE_KEYS = [
   "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"
]
LABEL_KEY = "Species"

_TRAIN_BATCH_SIZE = 14 # TODO check
_EVAL_BATCH_SIZE = 20 # TODO check

FEATURE_SPEC = {
    **{
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for feature in FEATURE_KEYS
    },
    LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
}


def _input_fn(
        file_pattern: List[str],
        data_accessor: tfx.components.DataAccessor,  # Input to record batch
        schema: schema_pb2.Schema,
        batch_size: int = 50
) -> tf.data.Dataset:  # (features: dict of tensors, indices: label indices)
   dataset = data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
         batch_size=batch_size,
         label_key=LABEL_KEY
      ),
      schema=schema
   )
   return dataset.repeat()


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
    schema = schema_utils.schema_from_feature_spec(FEATURE_SPEC)
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
        _TRAIN_BATCH_SIZE
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        schema,
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