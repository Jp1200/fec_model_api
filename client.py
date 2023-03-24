from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Define the neural network model and load the trained weights
def FECModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_size,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.load_weights('fec_model.h5')  # Load the trained weights from a file
    return model

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(lr=0.001)

# Define the loss function
def custom_loss(y_true, y_pred):
    # Calculate the loss as the absolute difference between the predicted FEC rate and the ground truth
    loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    return loss

# Initialize the model
input_size = 3  # The number of input features
model = FECModel()
model.compile(loss=custom_loss, optimizer=optimizer)

# Define the API routes
@app.route('/', methods=['GET'])
def index():
    return 'Welcome to the FEC optimizer API!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the input data from the user
    input_data = np.array(data['input_data'])  # Convert the input data to a NumPy array
    output_data = model.predict(input_data)  # Make a prediction using the model
    response = {'output_data': output_data.tolist()}  # Convert the output data to a list for JSON serialization
    return jsonify(response)

@app.route('/update', methods=['POST'])
def update():
    data = request.get_json()  # Get the input data from the user
    input_data = np.array(data['input_data'])  # Convert the input data to a NumPy array
    output_data = np.array(data['output_data'])  # Convert the output data to a NumPy array
    model.train_on_batch(input_data, output_data)  # Train the model on the new data
    return 'Model updated successfully!'

# Start the API server
if __name__ == '__main__':
    app.run()
