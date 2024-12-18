import numpy as np
import random
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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
    sequences = []
    inode_list = [inode for sublist in topology.values() for inode in sublist]

    for _ in range(num_sequences):
        sequence = []
        current_directory = random.choice(list(topology.keys()))
        for _ in range(sequence_length):
            files = topology[current_directory]
            if files:
                sequence.append(random.choice(files))
            current_directory = random.choice(list(topology.keys()))
        sequences.append(sequence)
    return sequences

# Step 3: Prepare data for XGBoost classifier with LabelEncoder
def prepare_xgb_classification_data(sequences, seq_length=10):
    X, y = [], []
    for seq in sequences:
        for i in range(len(seq) - seq_length):
            X.append(seq[i:i + seq_length])
            y.append(seq[i + seq_length])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Encode labels from 0

    return np.array(X), y_encoded, label_encoder

# Step 4: XGBoost Classifier Class
class XGBInodeClassifier:
    def __init__(self):
        # self.model = XGBClassifier(objective='multi:softmax', use_label_encoder=False, eval_metric='mlogloss')
        self.model = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', tree_method='gpu_hist')

        self.label_encoder = None

    def fit(self, X, y, label_encoder):
        self.label_encoder = label_encoder
        logger.info("Training XGBoost classifier...")
        self.model.fit(X, y, verbose=True)
        logger.info("XGBoost classifier training completed.")

    def predict(self, X):
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)  # Decode predictions

# Step 5: Evaluate Model and Visualize Results
def evaluate_and_visualize_classifier(predictor, X, y, label_encoder):
    predictions = predictor.predict(X)
    y_decoded = label_encoder.inverse_transform(y)  # Decode ground truth
    accuracy = accuracy_score(y_decoded, predictions)
    cm = confusion_matrix(y_decoded, predictions)

    logger.info(f"Classification Accuracy: {accuracy:.2%}")
    logger.info("Confusion Matrix:\n" + str(cm))

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_decoded)), y_decoded, label="Actual", marker="o")
    plt.plot(range(len(predictions)), predictions, label="Predicted", marker="x")
    plt.xlabel("Time Step")
    plt.ylabel("Inode IDs")
    plt.title("XGBoost Inode Classification")
    plt.legend()
    plt.savefig("inode_classification.png")

# Main Execution
if __name__ == "__main__":
    # Generate inode topology and access sequences
    inode_topology = generate_inode_topology(depth=3, breadth=5)
    logger.info(f"Inode topology: {inode_topology}")

    training_sequences = generate_logical_access_sequences(inode_topology, num_sequences=5, sequence_length=10000)
    testing_sequences = generate_logical_access_sequences(inode_topology, num_sequences=2, sequence_length=50)

    # Prepare data for XGBoost classification
    seq_length = 10
    X_train, y_train, label_encoder = prepare_xgb_classification_data(training_sequences, seq_length=seq_length)
    # X_test, y_test, _ = prepare_xgb_classification_data(testing_sequences, seq_length=seq_length)

    # get train data 0.8 and test data 0.2
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print(len(X_train))
    X_test = X_train[-3000:]
    y_test = y_train[-3000:]
    X_train = X_train[:-1000]
    y_train = y_train[:-1000]

    # Train XGBoost Classifier
    predictor = XGBInodeClassifier()
    predictor.fit(X_train, y_train, label_encoder)

    # Evaluate and visualize
    evaluate_and_visualize_classifier(predictor, X_test, y_test, label_encoder)
