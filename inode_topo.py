import logging
import random
import os


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
    Generates access sequences based on the inode topologenerate_access_sequences_from_topologygy.
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
    inode_topology, sequence_length=10
)

print(training_sequences)

# Step 3: Map sequences to HMM-compatible format
flat_sequences, lengths = prepare_data(training_sequences)

# Step 4: Train the HMM model with topology-based sequences
predictor = InodePredictor(num_states=len(state_set), num_inodes=num_inodes)
predictor.fit(flat_sequences, lengths)

# Step 5: Evaluate the model
accuracy = evaluate_model(predictor, testing_sequences)
logger.info(f"Model accuracy on testing data (top 3 predictions): {accuracy}")

# Visualize the predictions
visualize_predictions(predictor, testing_sequences)
