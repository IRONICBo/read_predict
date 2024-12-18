import numpy as np
from hmmlearn import hmm
import random
import logging
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt


# Initialize logging
def init_logger():
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


logger = init_logger()


# Step 1: Generate synthetic data for inode access sequences (1-200)
def generate_inode_data(sequence_length=1000, num_sequences=5, num_inodes=10):
    data = []
    for _ in range(num_sequences):
        sequence = [random.randint(1, num_inodes)]
        for _ in range(sequence_length - 1):
            next_inode = sequence[-1] + random.choice(
                [-1, 1]
            )  # Simulate sequential inode access
            if next_inode < 1:
                next_inode = 1
            elif next_inode > num_inodes:
                next_inode = num_inodes
            sequence.append(next_inode)
        data.append(sequence)
    return data


# Generate training and testing sequences
num_inodes = 200
training_sequences = generate_inode_data()
testing_sequences = generate_inode_data(sequence_length=100, num_sequences=2)


# Prepare data for HMM training
def prepare_data(sequences):
    lengths = [len(seq) for seq in sequences]
    flat_sequences = np.concatenate(sequences).reshape(-1, 1)
    return flat_sequences, lengths


flat_sequences, lengths = prepare_data(training_sequences)


# Step 1.5: Build a state set based on system instructions and file paths
def build_state_set(num_states=100):
    states = []
    for i in range(num_states):
        states.append(f"State_{i+1}")
    return states


state_set = build_state_set()
logger.info("State set constructed.")


# Step 2: Train a Hidden Markov Model
class InodePredictor:
    def __init__(self, num_states, num_inodes):
        self.num_states = 10
        # self.num_states = num_states
        self.num_inodes = num_inodes
        self.hmm = hmm.MultinomialHMM(
            n_components=num_states, n_iter=100, random_state=42
        )

    def fit(self, sequences, lengths):
        logger.info("Training HMM...")
        self.hmm.fit(sequences, lengths)
        logger.info("HMM training completed.")

    def predict_next_inodes(self, sequence, top_k=3):
        sequence = np.array(sequence).reshape(-1, 1)
        logprob, states = self.hmm.decode(sequence, algorithm="viterbi")
        last_state = states[-1]
        next_state_probs = self.hmm.transmat_[last_state]
        next_state = np.argmax(next_state_probs)
        next_emission_probs = self.hmm.emissionprob_[next_state]
        top_k_indices = np.argsort(next_emission_probs)[-top_k:][::-1]
        predicted_inodes = [idx + 1 for idx in top_k_indices]
        return predicted_inodes


# Instantiate and train the predictor
predictor = InodePredictor(num_states=len(state_set), num_inodes=num_inodes)
predictor.fit(flat_sequences, lengths)


# Evaluate model on testing data
def evaluate_model(predictor, testing_sequences, top_k=3):
    correct_predictions = 0
    total_predictions = 0

    for sequence in testing_sequences:
        for i in range(len(sequence) - 1):
            input_sequence = sequence[: i + 1]
            actual_next_inode = sequence[i + 1]
            predicted_inodes = predictor.predict_next_inodes(
                input_sequence, top_k=top_k
            )

            if actual_next_inode in predicted_inodes:
                correct_predictions += 1
            total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy


# Test prediction with a sample sequence
sample_sequence = [random.randint(1, num_inodes) for _ in range(10)]
predicted_inodes = predictor.predict_next_inodes(sample_sequence)
logger.info(f"Sample sequence: {sample_sequence}")
logger.info(f"Predicted next inodes (top 3): {predicted_inodes}")

# Evaluate the model
accuracy = evaluate_model(predictor, testing_sequences)
logger.info(f"Model accuracy on testing data (top 3 predictions): {accuracy}")


# Visualize predictions
def visualize_predictions(predictor, testing_sequences, num_days=50):
    actual = []
    predicted = []
    for sequence in testing_sequences:
        for i in range(min(num_days, len(sequence) - 1)):
            input_sequence = sequence[: i + 1]
            actual.append(sequence[i + 1])
            preds = predictor.predict_next_inodes(input_sequence, top_k=1)
            predicted.append(preds[0])

    plt.figure(figsize=(10, 6))
    plt.plot(actual, label="Actual", marker="o")
    plt.plot(predicted, label="Predicted", marker="x")
    plt.xlabel("Time Step")
    plt.ylabel("Inode")
    plt.title("Inode Access Prediction")
    plt.legend()
    plt.savefig("inode_access_prediction.png")


visualize_predictions(predictor, testing_sequences)

# Step 1: Simulate a directory structure for inode topology
def generate_inode_topology(root_path="/root", depth=3, breadth=5):
    """
    Simulates a directory structure for inode topology.
    :param root_path: Root directory path (string).
    :param depth: Depth of the directory tree.
    :param breadth: Number of files or subdirectories per directory.
    :return: Dictionary representing the topology and file paths.
    """
    inode_topology = {}
    inode_counter = 1  # Simulated inode IDs

    def create_subtree(path, current_depth):
        nonlocal inode_counter
        if current_depth > depth:
            return
        inode_topology[path] = []
        for i in range(breadth):
            if random.random() > 0.5:  # Randomly decide if it's a file or subdirectory
                # Add file
                file_path = f"{path}/file_{inode_counter}"
                inode_topology[path].append(inode_counter)
                inode_counter += 1
            else:
                # Add subdirectory
                subdir_path = f"{path}/dir_{inode_counter}"
                inode_topology[path].append(inode_counter)
                inode_counter += 1
                create_subtree(subdir_path, current_depth + 1)

    create_subtree(root_path, 1)
    return inode_topology


# Step 2: Generate synthetic inode access sequences based on the topology
def generate_access_sequences_from_topology(inode_topology, sequence_length=100):
    """
    Generates access sequences based on the inode topology.
    :param inode_topology: The topology dictionary.
    :param sequence_length: Length of access sequences to generate.
    :return: List of access sequences (list of lists).
    """
    access_sequences = []
    inode_list = [inode for sublist in inode_topology.values() for inode in sublist]

    for _ in range(sequence_length):
        sequence = []
        for _ in range(sequence_length):
            inode = random.choice(inode_list)
            sequence.append(inode)
        access_sequences.append(sequence)
    return access_sequences


# Simulate directory and inode structure
inode_topology = generate_inode_topology(depth=3, breadth=5)
logger.info(f"Inode topology constructed: {inode_topology}")

# Generate training and testing sequences from the topology
training_sequences = generate_access_sequences_from_topology(
    inode_topology, sequence_length=100
)
testing_sequences = generate_access_sequences_from_topology(
    inode_topology, sequence_length=100
)

# Step 3: Map sequences to HMM-compatible format
flat_sequences, lengths = prepare_data(training_sequences)

# Step 4: Train the HMM model with topology-based sequences
predictor = InodePredictor(num_states=len(state_set), num_inodes=num_inodes)
predictor.fit(flat_sequences, lengths)

# Step 5: Evaluate the model
accuracy = evaluate_model(predictor, training_sequences)
logger.info(f"Model accuracy on testing data (top 3 predictions): {accuracy}")

# Visualize the predictions
visualize_predictions(predictor, training_sequences)
