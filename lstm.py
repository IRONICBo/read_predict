import numpy as np
import random
import logging
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize logging
def init_logger():
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

logger = init_logger()

# Step 1: Simulate a directory structure for inode topology
def generate_inode_topology(root_path="/root", depth=3, breadth=5):
    """
    Simulates a directory structure for inode topology.
    :param root_path: Root directory path.
    :param depth: Depth of the directory tree.
    :param breadth: Number of files or subdirectories per directory.
    :return: Dictionary representing the topology and inode IDs.
    """
    inode_topology = {}
    inode_counter = 1

    def create_subtree(path, current_depth):
        nonlocal inode_counter
        if current_depth > depth:
            return
        inode_topology[path] = []
        for i in range(breadth):
            if random.random() > 0.5:  # Randomly decide file or subdirectory
                file_inode = inode_counter
                inode_topology[path].append(file_inode)
                inode_counter += 1
            else:
                subdir_path = f"{path}/dir_{inode_counter}"
                inode_topology[path].append(inode_counter)
                inode_counter += 1
                create_subtree(subdir_path, current_depth + 1)

    create_subtree(root_path, 1)
    return inode_topology

# Step 2: Generate logical access sequences from topology
def generate_logical_access_sequences(topology, num_sequences=10, sequence_length=100):
    """
    Generates logical access sequences based on the inode topology.
    Sequentially accesses files in each directory.
    """
    sequences = []
    inode_list = [inode for sublist in topology.values() for inode in sublist]

    for _ in range(num_sequences):
        sequence = []
        current_directory = random.choice(list(topology.keys()))
        for _ in range(sequence_length):
            files = topology[current_directory]
            if files:
                sequence.append(random.choice(files))  # Sequentially pick inode
            # Randomly switch directory
            current_directory = random.choice(list(topology.keys()))
        sequences.append(sequence)
    return sequences

# Step 3: Prepare data for LSTM model
def prepare_lstm_data(sequences, seq_length=10):
    """
    Prepare data for LSTM, scaling and padding sequences.
    :param sequences: List of access sequences.
    :param seq_length: Fixed sequence length for LSTM input.
    :return: X (input sequences), y (target values)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_data = np.concatenate(sequences).reshape(-1, 1)
    scaler.fit(all_data)

    X, y = [], []
    for seq in sequences:
        normalized_seq = scaler.transform(np.array(seq).reshape(-1, 1)).flatten()
        for i in range(len(normalized_seq) - seq_length):
            X.append(normalized_seq[i:i + seq_length])
            y.append(normalized_seq[i + seq_length])

    X = pad_sequences(X, maxlen=seq_length, dtype='float32')
    return np.array(X), np.array(y), scaler

# Step 4: LSTM Model Class
class LSTMInodePredictor:
    def __init__(self, input_shape):
        self.model = Sequential([
            LSTM(50, activation='tanh', input_shape=input_shape, return_sequences=False),
            Dense(1, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs=10, batch_size=32):
        logger.info("Training LSTM model...")
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        logger.info("LSTM model training completed.")

    def predict(self, X):
        return self.model.predict(X)

# Step 5: Evaluate Model and Visualize Results
def evaluate_and_visualize(predictor, X, y, scaler):
    predictions = predictor.predict(X)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y.reshape(-1, 1))

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(actual)), actual, label="Actual", marker="o")
    plt.plot(range(len(predictions)), predictions, label="Predicted", marker="x")
    plt.xlabel("Time Step")
    plt.ylabel("Inode Access")
    plt.title("LSTM Inode Access Prediction")
    plt.legend()
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Generate inode topology and access sequences
    inode_topology = generate_inode_topology(depth=3, breadth=5)
    logger.info(f"Inode topology: {inode_topology}")

    training_sequences = generate_logical_access_sequences(inode_topology, num_sequences=5, sequence_length=100)
    testing_sequences = generate_logical_access_sequences(inode_topology, num_sequences=2, sequence_length=50)

    # Prepare data for LSTM
    seq_length = 10
    X_train, y_train, scaler = prepare_lstm_data(training_sequences, seq_length=seq_length)
    X_test, y_test, _ = prepare_lstm_data(testing_sequences, seq_length=seq_length)

    # Train LSTM
    predictor = LSTMInodePredictor(input_shape=(X_train.shape[1], 1))
    predictor.fit(X_train, y_train, epochs=20, batch_size=32)

    # Evaluate and visualize
    evaluate_and_visualize(predictor, X_test, y_test, scaler)
