"""
    Code for training and evaluating bert or longformer
    Antonio Laverghetta Jr.
    Animesh Nighokjar
"""

from simpletransformers.classification import ClassificationModel
import pandas as pd
import numpy as np


# load the dataset as a pandas dataframe
# change the string argument to point to where your data is saved
data = pd.concat([pd.read_csv('satlib/uf50-218.csv', index_col=0), pd.read_csv('satlib/uuf50-218.csv', index_col=0)]).sample(frac=1).reset_index(drop=True)
print(data.head())

# split into train and test sets
split = np.random.rand(len(data)) < 0.8
train_df = data[split]
eval_df = data[~split]
print(train_df.shape, eval_df.shape)

# model_args sets various parameters for bert
# we reccommend leaving these at the default
# however, if you are running out memory when running the script, try lowering the batch size
model_args = {
    'eval_batch_size':32,
    'evaluate_during_training':True,
    'evaluate_during_training_steps':5000,
    'evaluate_during_training_verbose':True,
    'fp16': False, # needs apex
    'learning_rate':1e-8, # 4e-5, #(outputs)
    'max_seq_length':512, # 128, #(outputs)
    'n_gpu':1,
    'num_train_epochs':5,
    'overwrite_output_dir': False, # False if continuing training
    'output_dir': 'outputs-satlib',
    'reprocess_input_data': False, # False if continuing training
    'save_model_every_epoch': True,
    'silent': False,
    'train_batch_size':32,
    'save_best_model':True,
}

# initalize the bert model
model = ClassificationModel("bert", "bert-base-cased", args=model_args)

# uncomment line below and comment line above to use longformer
# model = ClassificationModel("longformer", "allenai/longformer-base-4096", args=model_args)

# Train the model
model.train_model(train_df, show_running_loss=True, eval_df=eval_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
print(result)