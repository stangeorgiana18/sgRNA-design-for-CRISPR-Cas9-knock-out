# Module for TensorFlow-based scoring 

"""
This script defines the CNN-based on-target regression model for CRISPR sgRNA efficiency prediction
using sequence-only features. It is intended to reproduce the exact architecture from the pre-trained 
model archive (ofttar_pt_cnn.tar.gz) provided in DeepCRISPR. 

Architecture summary:
  - Input: one-hot encoded sgRNA sequence of length 23 (shape: (23, 4))
  - Conv1D layer with 64 filters, kernel size 8, ReLU activation, 'same' padding
  - MaxPooling1D with pool size 2 (reduces the sequence length roughly by half)
  - Conv1D layer with 128 filters, kernel size 5, ReLU activation, 'same' padding
  - MaxPooling1D with pool size 2
  - Conv1D layer with 256 filters, kernel size 3, ReLU activation, 'same' padding
  - Flatten layer to convert 3D feature maps to 1D feature vector
  - Dense layer with 256 units and ReLU activation
  - Dropout layer (rate = 0.5) for regularization
  - Final Dense layer with 1 unit and linear activation for regression output

This model is expected to take as input a one-hot encoded sgRNA (23 nucleotides, A/C/G/T)
and output a single predicted efficiency score.

Usage:
    python deepcrispr_model.py <checkpoint_directory>

"""

import tensorflow as tf
import sys

class DeepCRISPRScorer:
    def __init__(self, checkpoint_dir):
        # Build the CNN architecture
        self.model = self._build_model()
        # Restore the pre-trained weights from the provided checkpoint directory.
        # The checkpoint should correspond exactly to the architecture defined below.
        checkpoint = tf.train.Checkpoint(model=self.model)
        latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_ckpt is None:
            raise ValueError("No checkpoint found in {}".format(checkpoint_dir))
        checkpoint.restore(latest_ckpt).expect_partial()
        print("Model successfully restored from:", latest_ckpt)

    def _build_model(self):
        """
        Constructs the CNN-based model architecture for sequence-only on-target regression.
        The architecture is defined to exactly match the pre-trained model (ontar_cnn_reg_seq).
        
        Returns:
            tf.keras.Model: The constructed CNN model.
        """
        model = tf.keras.Sequential([
            # Input layer expects a tensor of shape (23, 4)
            tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=8,
                strides=1,
                padding='same',
                activation='relu',
                input_shape=(30, 4), # input_shape=(23, 4),
                name='conv1'
            ),
            tf.keras.layers.MaxPooling1D(
                pool_size=2,
                strides=2,
                padding='same',
                name='pool1'
            ),
            tf.keras.layers.Conv1D(
                filters=128,
                kernel_size=5,
                strides=1,
                padding='same',
                activation='relu',
                name='conv2'
            ),
            tf.keras.layers.MaxPooling1D(
                pool_size=2,
                strides=2,
                padding='same',
                name='pool2'
            ),
            tf.keras.layers.Conv1D(
                filters=256,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu',
                name='conv3'
            ),
            tf.keras.layers.Flatten(name='flatten'),
            tf.keras.layers.Dense(256, activation='relu', name='fc1'),
            tf.keras.layers.Dropout(0.5, name='dropout'),
            tf.keras.layers.Dense(1, activation='linear', name='output')
        ], name="DeepCRISPR_Model")
        return model

    def one_hot_encode(self, sequence):
        """
        Converts a DNA sequence (string) of length 23 to a one-hot encoded tensor of shape (23, 4).
        Only accepts the characters A, C, G, and T.
        
        Args:
            sequence (str): sgRNA sequence (e.g., "ACGTACGTACGTACGTACGTACG")
        
        Returns:
            tf.Tensor: One-hot encoded tensor of shape (23, 4)
        """
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        if len(sequence) != 30:
            raise ValueError("Expected sequence length 30, got {}".format(len(sequence)))
        # Convert each nucleotide to its corresponding index
        seq_indices = [mapping[base] for base in sequence]
        # One-hot encode: result shape is (23, 4)
        one_hot = tf.one_hot(seq_indices, depth=4)
        return one_hot

    def predict_efficiency(self, sequences):
        """
        Predicts on-target efficiency scores for a list of sgRNA sequences.
        
        Args:
            sequences (list of str): List of sgRNA sequences (each 23 nucleotides long).
        
        Returns:
            list of float: Predicted efficiency scores.
        """
        # One-hot encode all sequences and stack them into a batch tensor of shape (batch_size, 23, 4)
        encoded_batch = tf.stack([self.one_hot_encode(seq) for seq in sequences])
        # Run prediction (in inference mode, training=False)
        predictions = self.model(encoded_batch, training=False)
        # Convert predictions to a flat Python list of floats
        return predictions.numpy().flatten().tolist()

# If this script is run directly, use an example sgRNA sequence for prediction.
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deepcrispr_model.py <checkpoint_directory>")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    scorer = DeepCRISPRScorer(checkpoint_dir)
    
    # Example sgRNA sequence (must be exactly 23 nucleotides long)
    example_sequence = "ACGTACGTACGTACGTACGTACG"
    efficiency = scorer.predict_efficiency([example_sequence])
    print("Predicted efficiency for {}: {:.4f}".format(example_sequence, efficiency[0]))

