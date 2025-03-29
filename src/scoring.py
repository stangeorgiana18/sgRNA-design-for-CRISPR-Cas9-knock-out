# Module for TensorFlow-based scoring 

"""
This script defines the CNN-based on-target regression model for CRISPR sgRNA efficiency prediction
using sequence-only features. It is intended to reproduce the exact architecture from the pre-trained 
model archive (ofttar_pt_cnn.tar.gz) provided in DeepSpCas9. 

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
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DeepSpCas9Scorer:
    def __init__(self, checkpoint_dir):
        self._build_graph()
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        # self.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))
        checkpoint_path = checkpoint_dir + "/PreTrain-Final-False-3-5-7-100-70-40-0.001-550-True-80-60"
        self.saver.restore(self.sess, checkpoint_path) # ensures loading the exact downloaded checkpoint
        print("DeepSpCas9 model successfully restored from:", checkpoint_path)
    
    def _build_graph(self):
        self.inputs = tf.placeholder(tf.float32, [None, 1, 30, 4])
        
        # Convolutional Layers
        self.conv1_W = tf.get_variable("conv1_W", shape=[1, 3, 4, 100])
        self.conv1_b = tf.get_variable("conv1_b", shape=[100])
        conv1 = tf.nn.conv2d(self.inputs, self.conv1_W, strides=[1,1,1,1], padding="VALID")
        conv1 = tf.nn.relu(conv1 + self.conv1_b)
        pool1 = tf.nn.avg_pool(conv1, ksize=[1,1,2,1], strides=[1,1,2,1], padding="SAME")
        
        self.conv2_W = tf.get_variable("conv2_W", shape=[1, 5, 4, 70])
        self.conv2_b = tf.get_variable("conv2_b", shape=[70])
        conv2 = tf.nn.conv2d(self.inputs, self.conv2_W, strides=[1,1,1,1], padding="VALID")
        conv2 = tf.nn.relu(conv2 + self.conv2_b)
        pool2 = tf.nn.avg_pool(conv2, ksize=[1,1,2,1], strides=[1,1,2,1], padding="SAME")
        
        self.conv3_W = tf.get_variable("conv3_W", shape=[1, 7, 4, 40])
        self.conv3_b = tf.get_variable("conv3_b", shape=[40])
        conv3 = tf.nn.conv2d(self.inputs, self.conv3_W, strides=[1,1,1,1], padding="VALID")
        conv3 = tf.nn.relu(conv3 + self.conv3_b)
        pool3 = tf.nn.avg_pool(conv3, ksize=[1,1,2,1], strides=[1,1,2,1], padding="SAME")
        
        # Flatten and Concatenate
        flatten0 = tf.reshape(pool1, [-1, 14 * 100])  # Adjust pooling output size
        flatten1 = tf.reshape(pool2, [-1, 13 * 70])
        flatten2 = tf.reshape(pool3, [-1, 12 * 40])
        concat = tf.concat([flatten0, flatten1, flatten2], axis=1)
        
        # Fully Connected Layers
        self.W_fcl1 = tf.get_variable("Fully_Connected_Layer1/W_fcl1", shape=[2790, 80])
        self.B_fcl1 = tf.get_variable("Fully_Connected_Layer1/B_fcl1", shape=[80])
        fc1 = tf.nn.relu(tf.matmul(concat, self.W_fcl1) + self.B_fcl1)
        
        self.W_fcl2 = tf.get_variable("Fully_Connected_Layer2/W_fcl2", shape=[80, 60])
        self.B_fcl2 = tf.get_variable("Fully_Connected_Layer2/B_fcl2", shape=[60])
        fc2 = tf.nn.relu(tf.matmul(fc1, self.W_fcl2) + self.B_fcl2)
        
        # Output Layer
        self.W_out = tf.get_variable("Output_Layer/W_out", shape=[60, 1])
        self.B_out = tf.get_variable("Output_Layer/B_out", shape=[1])
        self.outputs = tf.matmul(fc2, self.W_out) + self.B_out
    
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
        one_hot = tf.one_hot(seq_indices, depth=4)  # shape (30, 4)
        one_hot = tf.expand_dims(one_hot, axis=0)     # shape (1, 30, 4)
        return one_hot
    
    def predict_efficiency(self, sequences):
        """
    Predicts on-target efficiency scores for a list of sgRNA sequences.
    
    Args:
        sequences (list of str): List of sgRNA sequences (each 30 nucleotides long).
    
    Returns:
        numpy array: Predicted efficiency scores.
    """
        # Create a list of symbolic tensors, each of shape (1, 30, 4)
        tensor_list = [self.one_hot_encode(seq) for seq in sequences]
        # Evaluate the stacked tensor so that encoded_batch becomes a concrete NumPy array.
        encoded_batch = self.sess.run(tf.stack(tensor_list))
        # Run prediction in the session using the evaluated encoded_batch.
        predictions = self.sess.run(self.outputs, feed_dict={self.inputs: encoded_batch})

        return predictions.flatten().tolist() #  flatten the output predictions so that each candidate’s efficiency is a plain float


# If this script is run directly, use an example sgRNA sequence for prediction.
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deepcrispr_model.py <checkpoint_directory>")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    scorer = DeepSpCas9Scorer(checkpoint_dir)
    
    # Example sgRNA sequence (must be exactly 23 nucleotides long)
    example_sequence = "ACGTACGTACGTACGTACGTACG"
    efficiency = scorer.predict_efficiency([example_sequence])
    print("Predicted efficiency for {}: {:.4f}".format(example_sequence, efficiency[0]))

