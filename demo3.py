import numpy as np
from hmmlearn import hmm
import random


# Step 1: Generate synthetic data for inode access sequences (1-200)
def generate_inode_data(sequence_length=1000, num_sequences=5, num_inodes=200):
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
training_sequences = generate_inode_data()
testing_sequences = generate_inode_data(sequence_length=100, num_sequences=2)
num_inodes = 200


# Step 1.5: Build a state set based on system instructions and file paths
def build_state_set(num_states=100):
    states = []
    for i in range(num_states):
        states.append(f"State_{i+1}")
    return states


state_set = build_state_set()
print("State set:", state_set)

# Prepare data for HMM training
lengths = [len(seq) for seq in training_sequences]
flat_sequences = np.concatenate(training_sequences).reshape(
    -1, 1
)  # Convert to required shape

# Step 2: Train a Hidden Markov Model
model = hmm.MultinomialHMM(n_components=len(state_set), n_iter=100, random_state=42)
model.fit(flat_sequences, lengths)


# Step 3: Predict the next inode given a sequence (top 3 predictions)
def predict_next_inodes(model, sequence, num_inodes=200, top_k=3):
    sequence = np.array(sequence).reshape(-1, 1)
    logprob, states = model.decode(sequence, algorithm="viterbi")

    # Predict next inode based on transition probabilities from the last state
    last_state = states[-1]
    next_state_probs = model.transmat_[last_state]
    next_state = np.argmax(next_state_probs)
    next_emission_probs = model.emissionprob_[next_state]

    # Get top-k predicted inodes
    top_k_indices = np.argsort(next_emission_probs)[-top_k:][::-1]
    predicted_inodes = [
        idx + 1 for idx in top_k_indices
    ]  # Map back to inode range (1-200)

    return predicted_inodes


# Evaluate model on testing data
def evaluate_model(model, testing_sequences, top_k=3):
    correct_predictions = 0
    total_predictions = 0

    for sequence in testing_sequences:
        for i in range(len(sequence) - 1):
            input_sequence = sequence[: i + 1]
            actual_next_inode = sequence[i + 1]
            predicted_inodes = predict_next_inodes(model, input_sequence, top_k=top_k)

            if actual_next_inode in predicted_inodes:
                correct_predictions += 1
            total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy


# Test prediction with a sample sequence
sample_sequence = [random.randint(1, num_inodes) for _ in range(10)]
predicted_inodes = predict_next_inodes(model, sample_sequence)
print("Sample sequence:", sample_sequence)
print("Predicted next inodes (top 3):", predicted_inodes)

# Evaluate the model
accuracy = evaluate_model(model, testing_sequences)
print("Model accuracy on testing data (top 3 predictions):", accuracy)
