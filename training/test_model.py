import pickle
import numpy as np
from tensorflow import keras

model = keras.models.load_model("../model/student_model.keras")

with open("../model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

student = np.array([[6, 72, 85, 7, 8, 75]])
student_scaled = scaler.transform(student)

prediction = model.predict(student_scaled, verbose=0)[0][0]

print(f"Predicted Final Score: {prediction:.2f}")
