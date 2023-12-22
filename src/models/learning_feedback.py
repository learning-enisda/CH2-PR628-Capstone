import tensorflow as tf
import numpy as np

# this variable `times` is a numpy array containing the times users spent on the learning materials
times = 5

# Calculate the average time
average_time = np.mean(times)

# Define a simple model with one input (the time a user spent on a learning material)
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model based on data
model.fit(times, times, epochs=10)

def provide_feedback(user_time):
  expected_time = model.predict(np.array([user_time]))[0][0]

  if user_time < expected_time * 0.5:
    return "Woww kamu menyelesaikan materi ini dengan sangat cepat. Ingatlah, belajar bahasa isyarat membutuhkan perhatian terhadap hal-hal yang detail. Cobalah untuk meluangkan lebih banyak waktu untuk setiap materi untuk memastikan kamu tidak melewatkan apapun"
  else:
    return "Kerja baguss!"
