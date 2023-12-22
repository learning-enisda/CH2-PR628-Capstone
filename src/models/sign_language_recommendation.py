import numpy as np
import tensorflow as tf

signs = [1, 6, 11, 16, 21]
next_signs = [2, 7, 12, 17, 22]

reader = 1  # The user who is reading the material or studying

# Convert your lists to numpy arrays
signs_array = np.array(signs)
next_signs_array = np.array(next_signs)

# this variabel num_signs is the total number of unique signs
num_signs = 26  # Total number of signs

# create array with the same shape as signs_array filled with the reader value
reader_array = np.full_like(signs_array, reader)

# this to combine reader and signs into a single input array
combined_input = np.column_stack((reader_array, signs_array))

# Build model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_signs, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy') # I think we can use another optimizer

# Train model
model.fit(combined_input, next_signs_array, epochs=700)

# Save the trained model to h5
model.save('sign_language_recommendation_model.h5')
