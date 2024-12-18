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


# Generate training sequences
sequences = generate_inode_data()
num_inodes = 200

# Prepare data for HMM training
lengths = [len(seq) for seq in sequences]
flat_sequences = np.concatenate(sequences).reshape(-1, 1)  # Convert to required shape

# Step 2: Train a Hidden Markov Model
model = hmm.MultinomialHMM(n_components=5, n_iter=100, random_state=42)
model.fit(flat_sequences, lengths)


# Step 3: Predict the next inode given a sequence
def predict_next_inode(model, sequence, num_inodes=200):
    sequence = np.array(sequence).reshape(-1, 1)
    logprob, states = model.decode(sequence, algorithm="viterbi")

    # Predict next inode based on transition probabilities from the last state
    last_state = states[-1]
    next_state_probs = model.transmat_[last_state]
    next_state = np.argmax(next_state_probs)
    next_emission_probs = model.emissionprob_[next_state]
    predicted_inode = (
        np.argmax(next_emission_probs) + 1
    )  # Map back to inode range (1-200)

    return predicted_inode


# Test prediction
sample_sequence = [
    random.randint(1, num_inodes) for _ in range(10)
]  # Generate a random sequence
predicted_inode = predict_next_inode(model, sample_sequence)

print("Sample sequence:", sample_sequence)
print("Predicted next inode:", predicted_inode)
