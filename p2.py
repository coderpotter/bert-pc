from simpletransformers.classification import ClassificationModel
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
# import wandb
import numpy as np
# import torch

# wandb.login()
# wandb.init(
#     project="bert-pc",
#     config={
#         "learning_rate": 1e-8,
#         "architecture": "bert",
#         "dataset": "antonio",
#         "batch_size": 16,
#         }
# )
# config = wandb.config

# # logging.basicConfig(level=logging.INFO)
# # transformers_logger = logging.getLogger("transformers")
# # transformers_logger.setLevel(logging.WARNING)

data = pd.concat([pd.read_csv('satlib/uf50-218.csv', index_col=0), pd.read_csv('satlib/uuf50-218.csv', index_col=0)]).sample(frac=1).reset_index(drop=True)
# eval_df = pd.concat([pd.read_csv('test/grp1/test.csv', index_col=0), pd.read_csv('test/grp2/test.csv', index_col=0)])
print(data.head())

split = np.random.rand(len(data)) < 0.8
train_df = data[split]
eval_df = data[~split]
print(train_df.shape, eval_df.shape)
# train_texts = train_df['S-exp'].to_list()
# train_labels = train_df['labels'].astype(np.float16).to_list()
# eval_texts = eval_df['S-exp'].to_list()
# eval_labels = eval_df['labels'].astype(np.float16).to_list()

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# print('encoding train data')
# train_encodings = tokenizer(train_texts, truncation=True, padding=True)
# print('encoding eval data')
# eval_encodings = tokenizer(eval_texts, truncation=True, padding=True)

# def SATDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# train_dataset = SATDataset(train_encodings, train_labels)
# eval_dataset = SATDataset(eval_encodings, eval_labels)

# model = BertForSequenceClassification.from_pretrained(
#     "bert-base-cased",
#     num_labels=2
# )
# model.cuda()

# training_args = TrainingArguments(
#     output_dir = 'outputs1',
#     # overwrite_output_dir = False, # (defaults to False) If True, overwrite the content of the output directory. Use this to continue training if output_dir points to a checkpoint directory.
#     do_train = True,
#     do_eval = True,
#     do_predict = False,
#     evaluate_during_training = True,
#     evaluation_strategy = 'epochs', # (defaults to "no") - The evaluation strategy to adopt during training. Possible values are:  "no": No evaluation is done during training.  "steps": Evaluation is done (and logged) every eval_steps.  "epoch": Evaluation is done at the end of each epoch.
#     # prediction_loss_only = False, # (defaults to False) - When performing evaluation and predictions, only returns the loss.
#     per_device_train_batch_size = 16, # (defaults to 8) - The batch size per GPU/TPU core/CPU for training.
#     per_device_eval_batch_size = 16, # (defaults to 8) - The batch size per GPU/TPU core/CPU for evaluation.
#     # per_gpu_train_batch_size = None,
#     # per_gpu_eval_batch_size = None,
#     # gradient_accumulation_steps = 1, # (defaults to 1) - Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
#     eval_accumulation_steps = 1, # Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If left None, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster but requires more memory).
#     learning_rate = 1e-08, # (defaults to 5e-5) - The initial learning rate for Adam.
#     # weight_decay = 0.0, # (defaults to 0) - The weight decay to apply (if not zero).
#     # adam_beta1 = 0.9, # (defaults to 0.9) – Beta1 for the Adam optimizer.
#     # adam_beta2 = 0.999, # (defaults to 0.999) – Beta2 for the Adam optimizer.
#     # adam_epsilon = 1e-08, # (defaults to 1e-08) – Epsilon for the Adam optimizer.
#     # max_grad_norm = 1.0, # (defaults to 1.0) – Maximum gradient norm (for gradient clipping).
#     num_train_epochs = 5.0, # (defaults to 3.0) – Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).
#     # max_steps = -1, # (defaults to -1) – If set to a positive number, the total number of training steps to perform. Overrides num_train_epochs.
#     warmup_steps = 0, # (defaults to 0) – Number of steps used for a linear warmup from 0 to learning_rate.
#     # logging_dir, # Tensorboard log directory. Will default to runs/**CURRENT_DATETIME_HOSTNAME**.
#     logging_first_step = True, # (defaults to False) – Whether to log and evaluate the first global_step or not.
#     # logging_steps = 500, # (defaults to 500) – Number of update steps between two logs.
#     save_steps = 5000, # (defaults to 500) – Number of updates steps before two checkpoint saves.
#     # save_total_limit = None, # If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir.
#     # no_cuda = False, # (defaults to False) – Whether to not use CUDA even when it is available or not.
#     # seed = 42, # (defaults to 42) – Random seed for initialization.
#     fp16 = True, # (defaults to False) – Whether to use 16-bit (mixed) precision training (through NVIDIA apex) instead of 32-bit training.
#     # fp16_opt_level = 'O1', # (defaults to ‘O1’) – For fp16 training, apex AMP optimization level selected in [‘O0’, ‘O1’, ‘O2’, and ‘O3’]. See details on the apex documentation.
#     # local_rank = -1, # (defaults to -1) - During distributed training, the rank of the process.
#     # tpu_num_cores = None, # When training on TPU, the number of TPU cores (automatically passed by launcher script).
#     # tpu_metrics_debug = False,
#     # debug = False, # (defaults to False) - When training on TPU, whether to print debug metrics or not.
#     # dataloader_drop_last = False, # (defaults to False) - Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size) or not.
#     eval_steps = 2000, # Number of update steps between two evaluations if evaluation_strategy="steps". Will default to the same value as logging_steps if not set.
#     # dataloader_num_workers = 0, # (defaults to 0) – Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process.
#     # past_index = -1, # (defaults to -1) – Some models like TransformerXL or :doc`XLNet <../model_doc/xlnet>` can make use of the past hidden states for their predictions. If this argument is set to a positive int, the Trainer will use the corresponding output (usually index 2) as the past state and feed it to the model at the next training step under the keyword argument mems.
#     # run_name = None, # A descriptor for the run. Notably used for wandb logging.
#     disable_tqdm = False, # Whether or not to disable the tqdm progress bars. Will default to True if the logging level is set to warn or lower (default), False otherwise.
#     # remove_unused_columns = True, # (defaults to True) – If using nlp.Dataset datasets, whether or not to automatically remove the columns unused by the model forward method.
#     label_names = ['SAT'], # The list of keys in your dictionary of inputs that correspond to the labels. Will eventually default to ["labels"] except if the model used is one of the XxxForQuestionAnswering in which case it will default to ["start_positions", "end_positions"].
#     # load_best_model_at_end = False, # (defaults to False) – Whether or not to load the best model found during training at the end of training.
#     # metric_for_best_model = None, # Use in conjunction with load_best_model_at_end to specify the metric to use to compare two different models. Must be the name of a metric returned by the evaluation with or without the prefix "eval_". Will default to "loss" if unspecified and load_best_model_at_end=True (to use the evaluation loss). If you set this value, greater_is_better will default to True. Don’t forget to set it to False if your metric is better when lower.
#     # greater_is_better = None # Use in conjunction with load_best_model_at_end and metric_for_best_model to specify if better models should have a greater metric or not. Will default to: True if metric_for_best_model is set to a value that isn’t "loss" or "eval_loss". False if metric_for_best_model is not set, or set to "loss" or "eval_loss".
# )

# trainer = Trainer(
#     model = model, # The model to train, evaluate or use for predictions. If not provided, a model_init must be passed.
#     args = training_args, # The arguments to tweak for training. Will default to a basic instance of TrainingArguments with the output_dir set to a directory named tmp_trainer in the current directory if not provided.
#     # data_collator = None, # The function to use to form a batch from a list of elements of train_dataset or eval_dataset. Will default to default_data_collator() if no tokenizer is provided, an instance of DataCollatorWithPadding() otherwise.
#     train_dataset = train_df, # The dataset to use for training. If it is an datasets.Dataset, columns not accepted by the model.forward() method are automatically removed.
#     eval_dataset = eval_df,
#     # tokenizer = tokenizer, # The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an interrupted training or reuse the fine-tuned model.
#     # model_init = None, # A function that instantiates the model to be used. If provided, each call to train() will start from a new instance of the model as given by this function. The function may have zero argument, or a single one containing the optuna/Ray Tune trial object, to be able to choose different architectures according to hyper parameters (such as layer count, sizes of inner layers, dropout probabilities etc).
#     # compute_metrics = None, # The function that will be used to compute metrics at evaluation. Must take a EvalPrediction and return a dictionary string to metric values.
#     # callbacks = None, # A list of callbacks to customize the training loop. Will add those to the list of default callbacks detailed in here. If you want to remove one of the default callbacks used, use the Trainer.remove_callback() method.
#     # optimizers = (None, None), # A tuple containing the optimizer and the scheduler to use. Will default to an instance of AdamW on your model and a scheduler given by get_linear_schedule_with_warmup() controlled by args.
# )

# trainer.train()
# trainer.evaluate()

model_args = {
    'eval_batch_size':32,
    'evaluate_during_training':True,
    'evaluate_during_training_steps':5000,
    'evaluate_during_training_verbose':True,
    'fp16': False, # needs apex
    'learning_rate':1e-8, # 4e-5, #(outputs)
    'max_seq_length':512, # 128, #(outputs)
    'n_gpu':3,
    'num_train_epochs':5,
    'overwrite_output_dir': False, # False if continuing training
    'output_dir': 'outputs-satlib',
    'reprocess_input_data': False, # False if continuing training
    'save_model_every_epoch': True,
    'silent': False,
    'train_batch_size':32,
    'save_best_model':True,
}

# Create a ClassificationModel
model = ClassificationModel("bert", "bert-base-cased", args=model_args)
# model = ClassificationModel("longformer", "allenai/longformer-base-4096", args=model_args)

# Train the model
model.train_model(train_df, show_running_loss=True, eval_df=eval_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
print(result)

# wandb.finish()