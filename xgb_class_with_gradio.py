import gradio as gr
import numpy as np
import random
import time
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

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

# Step 1: Generate inode topology
def generate_inode_topology(root_path="/root", depth=3, breadth=5):
    inode_topology = {}
    inode_counter = 1

    def create_subtree(path, current_depth):
        nonlocal inode_counter
        if current_depth > depth:
            return
        inode_topology[path] = []
        for i in range(breadth):
            if random.random() > 0.5:
                inode_topology[path].append(inode_counter)
                inode_counter += 1
            else:
                subdir_path = f"{path}/dir_{inode_counter}"
                inode_topology[path].append(inode_counter)
                inode_counter += 1
                create_subtree(subdir_path, current_depth + 1)

    create_subtree(root_path, 1)
    return inode_topology

# Step 2: Generate logical access sequences
def generate_logical_access_sequences(topology, num_sequences=10, sequence_length=100):
    sequences = []
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

# Step 3: Prepare data for XGBoost classification
def prepare_xgb_classification_data(sequences, seq_length=10):
    X, y = [], []
    for seq in sequences:
        for i in range(len(seq) - seq_length):
            X.append(seq[i:i + seq_length])
            y.append(seq[i + seq_length])
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return np.array(X), y_encoded, label_encoder

# Step 4: XGBoost Classifier Class
class XGBInodeClassifier:
    def __init__(self):
        self.model = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', tree_method='hist', device='cuda')

    def fit(self, X, y, label_encoder):
        self.label_encoder = label_encoder
        start_time = time.time()
        self.model.fit(X, y)
        end_time = time.time()
        return end_time - start_time  # Return training time

    def predict(self, X):
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)

# Step 5: Evaluate Model and Visualize Results
def evaluate_and_visualize_classifier(predictor, X, y, label_encoder):
    predictions = predictor.predict(X)
    y_decoded = label_encoder.inverse_transform(y)
    accuracy = accuracy_score(y_decoded, predictions)
    cm = confusion_matrix(y_decoded, predictions)

    # Confusion Matrix
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Trend Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_decoded)), y_decoded, label="Actual", marker="o")
    plt.plot(range(len(predictions)), predictions, label="Predicted", marker="x")
    plt.xlabel("Time Step")
    plt.ylabel("Inode IDs")
    plt.title("Prediction vs Actual")
    plt.legend()
    plt.savefig("prediction_trend.png")
    plt.close()

    return accuracy, "confusion_matrix.png", "prediction_trend.png"

# Gradio App
def gradio_interface(depth, breadth, seq_length, num_sequences, test_size):
    # Step 1: Generate data
    topology = generate_inode_topology(depth=depth, breadth=breadth)
    sequences = generate_logical_access_sequences(topology, num_sequences=num_sequences, sequence_length=seq_length)

    # Step 2: Prepare data
    X, y, label_encoder = prepare_xgb_classification_data(sequences, seq_length=10)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[int(split_idx*0.8):]
    y_train, y_test = y[:split_idx], y[int(split_idx*0.8):]

    # Step 3: Train model
    predictor = XGBInodeClassifier()
    train_time = predictor.fit(X_train, y_train, label_encoder)

    # Step 4: Evaluate and visualize
    accuracy, cm_path, trend_path = evaluate_and_visualize_classifier(predictor, X_test, y_test, label_encoder)

    # Return results
    return f"Accuracy: {accuracy:.2%}\nTraining Time: {train_time:.2f} seconds", cm_path, trend_path

# Gradio UI
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Slider(2, 5, step=1, label="Directory Depth"),
        gr.Slider(3, 10, step=1, label="Directory Breadth"),
        gr.Slider(50, 500, step=50, label="Sequence Length"),
        gr.Slider(5, 50, step=5, label="Number of Sequences"),
        gr.Slider(0.1, 0.5, step=0.1, label="Test Size Ratio"),
    ],
    outputs=[
        "text",  # Accuracy and training time
        "image",  # Confusion matrix
        "image"   # Prediction trend
    ],
    title="Inode Prediction Model",
    description="Test and visualize an XGBoost-based inode prediction model with interactive inputs."
)

iface.launch()
