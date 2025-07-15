import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pathlib
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
IMAGE_DIR = "bird_images"
OUTPUT_GRAPH = "output_graph3.pb"
OUTPUT_LABELS = "output_labels3.txt"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

def main():
    data_dir = pathlib.Path(IMAGE_DIR)
    class_names = sorted([item.name for item in data_dir.glob("*") if item.is_dir()])
    
    # Save class labels
    with open(OUTPUT_LABELS, "w") as f:
        for label in class_names:
            f.write(label + "\n")

    # Image loading
    datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1. / 255
    )

    train_data = datagen.flow_from_directory(
        IMAGE_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        IMAGE_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        subset='validation'
    )

    # Load base model
    base_model = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5",
                                input_shape=IMAGE_SIZE + (3,), trainable=False)

    model = tf.keras.Sequential([
        base_model,
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    model.fit(train_data, epochs=EPOCHS, validation_data=val_data)

    # Save model as .pb
    tf.saved_model.save(model, "saved_model")

    # Convert to .pb frozen graph
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(shape=(None,) + IMAGE_SIZE + (3,), dtype=tf.float32))

    frozen_func = tf.graph_util.convert_variables_to_constants_v2(full_model)
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=".",
                      name=OUTPUT_GRAPH,
                      as_text=False)

    print("Model saved to", OUTPUT_GRAPH)
    print("Labels saved to", OUTPUT_LABELS)

if __name__ == "__main__":
    main()
