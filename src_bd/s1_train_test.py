
if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    print(ROOT)
import numpy as np
import torch

if 1: # my lib
    import utils.lib_io as lib_io
    import utils.lib_commons as lib_commons
    import utils.lib_datasets as lib_datasets
    import utils.lib_augment as lib_augment
    import utils.lib_ml as lib_ml
    import utils.lib_rnn as lib_rnn
# Init env
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()
print(torch.cuda.is_available())
torch.cuda.current_device()
torch.cuda._initialized = True

# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------


# Set arguments ------------------------- 
args = lib_rnn.set_default_args()

args.learning_rate = 0.001
args.num_epochs = 20
args.batch_size = 64
args.learning_rate_decay_interval = 5 # decay for every 3 epochs
args.learning_rate_decay_rate = 0.5 # lr = lr * rate
args.train_eval_test_ratio=[0.9, 0.1, 0.0]
args.data_folder = "../data/speechv1_10/data_train/"
args.classes_txt = "../config/classes_10.names"
#test
args.test_data_folder = "../data/speechv1_10/data_test/"
args.save_model_to = '../checkpoints/normal_10/'
args.load_weights_from = None# #None:trian True:test
if not args.load_weights_from==None:
    args.load_weights_from = args.save_model_to+'020.ckpt'
# Dataset -------------------------- 

# Get data's filenames and labels
files_name, files_label = lib_datasets.AudioDataset.load_filenames_and_labels(
    args.data_folder, args.classes_txt)

# Split data into train/eval/test
tr_X, tr_Y, ev_X, ev_Y, _, _ = lib_ml.split_train_eval_test(
    X=files_name, Y=files_label, ratios=args.train_eval_test_ratio, dtype='list')
train_dataset = lib_datasets.AudioDataset(files_name=tr_X, files_label=tr_Y, transform=None)
eval_dataset = lib_datasets.AudioDataset(files_name=ev_X, files_label=ev_Y, transform=None)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=True)

# Create model and train -------------------------------------------------
model = lib_rnn.create_RNN_model(args, load_weights_from=args.load_weights_from) # create model
if args.load_weights_from is None:
    lib_rnn.train_model(model, args, train_loader, eval_loader)
else:
# test saved model
    testfiles_name, files_label = lib_datasets.AudioDataset.load_filenames_and_labels(
        args.test_data_folder, args.classes_txt)
    test_dataset = lib_datasets.AudioDataset(files_name=testfiles_name, files_label=files_label, transform=None)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
    lib_rnn.evaluate_model(model,test_loader)
