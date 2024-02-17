
import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess(image, label):
    """
    Normalizes images: `uint8` -> `float32`.
    """
    return tf.cast(image, tf.float32) / 255., label

def prepare_datasets(split_ratios=(70, 15, 15), batch_size=128):
    """
    Prepares training, validation, and test datasets from TensorFlow Datasets.
    Args:
    - split_ratios: A tuple of integers representing the percentage split for training, validation, and test datasets.
    - batch_size: The size of each batch for training and evaluation.

    Returns:
    - A tuple containing the training, validation, and test datasets.
    """
    assert sum(split_ratios) == 100, "Split ratios must sum to 100."

    # Load the dataset
    ds_train, ds_info = tfds.load('cifar10', split='train', as_supervised=True, with_info=True)

    # Calculate the sizes of each split
    train_size = ds_info.splits['train'].num_examples
    train_split = split_ratios[0] * train_size // 100
    val_split = split_ratios[1] * train_size // 100

    # Split the dataset
    ds_train_split = ds_train.take(train_split)
    ds_val_split = ds_train.skip(train_split).take(val_split)
    ds_test_split = ds_train.skip(train_split + val_split)

    # Preprocess and batch the datasets
    ds_train_processed = ds_train_split.map(preprocess).cache().shuffle(10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    ds_val_processed = ds_val_split.map(preprocess).batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    ds_test_processed = ds_test_split.map(preprocess).batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train_processed, ds_val_processed, ds_test_processed, ds_info

# Prepare the datasets
ds_train, ds_validation, ds_test, ds_info = prepare_datasets()

print("Datasets loaded and preprocessed:")
print("Training set:", ds_train)
print("Validation set:", ds_validation)
print("Testing set:", ds_test)
