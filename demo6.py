import numpy as np
import random
import logging
from hmmlearn import hmm
import matplotlib.pyplot as plt

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

# Step 3: Normalize inode sequences to fit HMM format
def normalize_inode_sequence(sequence, num_inodes=200):
    min_val = min(sequence)
    return [(inode - min_val) % num_inodes for inode in sequence]

def prepare_data(sequences):
    lengths = [len(seq) for seq in sequences]
    flat_sequences = np.concatenate(sequences).reshape(-1, 1)
    return flat_sequences, lengths

# Step 4: HMM Model Class
class InodePredictor:
    def __init__(self, num_states):
        self.num_states = num_states
        self.hmm = hmm.MultinomialHMM(n_components=num_states, n_iter=100, random_state=42)

    def fit(self, sequences, lengths):
        logger.info('Training HMM...')
        self.hmm.fit(sequences, lengths)
        logger.info('HMM training completed.')

    def predict_next_inodes(self, sequence, top_k=3):
        sequence = np.array(sequence).reshape(-1, 1)
        _, states = self.hmm.decode(sequence, algorithm="viterbi")
        last_state = states[-1]
        next_emission_probs = self.hmm.emissionprob_[last_state]
        top_k_indices = np.argsort(next_emission_probs)[-top_k:][::-1]
        predicted_inodes = [idx for idx in top_k_indices]
        return predicted_inodes

# Step 5: Evaluate Model
def evaluate_model(predictor, testing_sequences, top_k=3):
    correct_predictions = 0
    total_predictions = 0

    for sequence in testing_sequences:
        for i in range(len(sequence) - 1):
            input_sequence = sequence[:i+1]
            actual_next_inode = sequence[i+1]
            predicted_inodes = predictor.predict_next_inodes(input_sequence, top_k=top_k)
            if actual_next_inode in predicted_inodes:
                correct_predictions += 1
            total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

# Step 6: Visualization
def visualize_predictions(predictor, testing_sequences, num_days=50):
    actual = []
    predicted = []

    for sequence in testing_sequences:
        for i in range(min(num_days, len(sequence) - 1)):
            input_sequence = sequence[:i+1]
            actual.append(sequence[i+1])
            preds = predictor.predict_next_inodes(input_sequence, top_k=1)
            predicted.append(preds[0])

    plt.figure(figsize=(10, 6))
    plt.plot(actual, label="Actual", marker="o")
    plt.plot(predicted, label="Predicted", marker="x")
    plt.xlabel("Time Step")
    plt.ylabel("Inode")
    plt.title("Inode Access Prediction")
    plt.legend()
    plt.savefig("inode_access_prediction1.png")

# Main Execution
if __name__ == "__main__":
    # Generate inode topology and access sequences
    inode_topology = generate_inode_topology(depth=3, breadth=5)
    logger.info(f"Inode topology: {inode_topology}")

    training_sequences = generate_logical_access_sequences(inode_topology, num_sequences=5, sequence_length=100)
    testing_sequences = generate_logical_access_sequences(inode_topology, num_sequences=2, sequence_length=50)

    # Normalize sequences
    normalized_training_sequences = [normalize_inode_sequence(seq) for seq in training_sequences]
    normalized_testing_sequences = [normalize_inode_sequence(seq) for seq in testing_sequences]

    # Prepare data
    flat_sequences, lengths = prepare_data(normalized_training_sequences)

    # Train HMM
    predictor = InodePredictor(num_states=10)
    predictor.fit(flat_sequences, lengths)

    # Evaluate
    accuracy = evaluate_model(predictor, normalized_testing_sequences)
    logger.info(f"Model accuracy on testing data (top 3 predictions): {accuracy}")

    # Visualize predictions
    visualize_predictions(predictor, normalized_testing_sequences)
