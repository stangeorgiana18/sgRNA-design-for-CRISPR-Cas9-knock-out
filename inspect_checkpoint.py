# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# print_tensors_in_checkpoint_file(
#     file_name="PreTrain-Final-False-3-5-7-100-70-40-0.001-550-True-80-60",
#     tensor_name="",
#     all_tensors=False
# )

# Total number of params: 696395

import tensorflow as tf

# list all the variable names stored in your checkpoint
# If you need TF1 behavior, disable TF2 execution:
tf.compat.v1.disable_eager_execution()

checkpoint_path = "src/models/PreTrain-Final-False-3-5-7-100-70-40-0.001-550-True-80-60"  # Use the checkpoint prefix (without file extension)

reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print(key, var_to_shape_map[key])
    
# Command to run this script: python inspect_checkpoint.py
