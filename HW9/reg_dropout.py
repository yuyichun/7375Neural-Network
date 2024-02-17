import tensorflow as tf
from tensorflow.keras import layers, regularizers

# Define the neural network with L2 regularization and dropout


def build_model(input_shape, num_classes, dropout_rate=0.2, l2_penalty=0.001):
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_penalty)),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_penalty)),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# Define constants
input_shape = (28, 28)
num_classes = 10
epochs = 10
batch_size = 32
dropout_rate = 0.2
l2_penalty = 0.001

# Build the model
model = build_model(input_shape, num_classes, dropout_rate, l2_penalty)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
