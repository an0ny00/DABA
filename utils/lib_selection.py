if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    print(ROOT)
import numpy as np
import torch
import math
import pickle as pkl
import torch.nn.functional as F
from collections import Counter
import torchextractor as tx
import glob
import random
if 1: # my lib
    import utils.lib_io as lib_io
    import utils.lib_commons as lib_commons
    import utils.lib_datasets as lib_datasets
    import utils.lib_augment as lib_augment
    import utils.lib_ml as lib_ml
    import utils.lib_rnn as lib_rnn
    import utils.lib_tool as lib_tool
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


def calc_ent(X):
        """
        H(X) = -sigma p(x)log p(x)
        :param X:
        :return:
        """
        # x_values = {}
        # for x in X:
        #     x_values[x] = x_values.get(x, 0) + 1
        # length = len(x_values)
        ans = 0
        for p in X:
            # p = x_values.get(x) / length
            ans += p * math.log2(p)

        return 0 - ans

def one_sotamax_entropy(model,audio_path):
    audio = lib_datasets.AudioClass(filename=audio_path)
    audio.compute_mfcc()
    x = audio.mfcc.T
    x = torch.tensor(x[np.newaxis, :], dtype=torch.float32)
    x = x.to(model.device)
    output = model.forward(x)
    sf = F.softmax(output.data,dim=1)
    sf_ = sf.cpu().numpy().tolist()[0]
    se = calc_ent(sf_)
    # a, predicted = torch.max(output.data, 1)
    return sf.cpu().numpy()[0], se

def Cer_sotamax_entropy(model,trigger_pool): #computer certainty
    trigger_names = lib_commons.get_filenames(trigger_pool, "*.wav")
    se_list = []
    for trigger_path in trigger_names:
        _,se = one_sotamax_entropy(model, trigger_path)
        se_list.append(se)
    Cer_dict = dict(zip(trigger_names,se_list))
    if not os.path.exists('../data/dict/Cer.pkl'):
        with open('../data/dict/Cer.pkl', 'ab') as f:
            pkl.dump(Cer_dict, f)
    return Cer_dict

def Cer_triggers_selection(model,trigger_pool,rank):
    rank-=1
    if os.path.exists('../data/dict/Cer.pkl'):
        base_dict = pkl.load(open('../data/dict/Cer.pkl', 'rb'))
    else:
        base_dict = Cer_sotamax_entropy(model,trigger_pool)
    d_order_frommax = sorted(base_dict.items(), key=lambda x: x[1], reverse=True)
    d_order_frommin = sorted(base_dict.items(), key=lambda x: x[1], reverse=False)
    return d_order_frommax[rank],d_order_frommin[rank]
#
def cross_entropy(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

def Inf_cross_entropy(model,trigger_path,hosts_path,po_db=-20): #computer influence
    entropy_list = []
    if isinstance(hosts_path,list):
        host_samples_path = hosts_path
    else:
        host_samples_path = lib_commons.get_filenames(hosts_path, "*.wav")
    for host_path in host_samples_path:
        _,poison_path = lib_tool.Single_trigger_injection(org_wav_path=host_path,trigger_wav_path=trigger_path,output_path='../data/trigger_pool/cut_music.wav',po_db=po_db)

        trigger_sf,_ = one_sotamax_entropy(model,trigger_path)
        poison_sf,_ = one_sotamax_entropy(model,poison_path)
        one_ce = cross_entropy(trigger_sf,poison_sf)

        entropy_list.append(one_ce)
    Inf_hosts = dict(zip(host_samples_path, entropy_list))
    if not os.path.exists('../data/dict/Inf_hosts.pkl'):
        with open('../data/dict/Inf_hosts.pkl', 'ab') as f:
            pkl.dump(Inf_hosts, f)
    return Inf_hosts

def Inf_hosts_selection(model,trigger_path,hosts_path,po_nums):
    if os.path.exists('../data/dict/Inf_hosts.pkl'):
        base_dict = pkl.load(open('../data/dict/Inf_hosts.pkl', 'rb'))
    else:
        base_dict = Inf_cross_entropy(model,trigger_path,hosts_path)
    d_order_frommin = sorted(base_dict.items(), key=lambda x: x[1], reverse=False)
    d_order_fromax = sorted(base_dict.items(), key=lambda x: x[1], reverse=True)

    d_order_fromax_list = [i[0] for i in d_order_fromax]
    d_order_frommin_list = [i[0] for i in d_order_frommin]

    return d_order_fromax_list[:po_nums], d_order_frommin_list[:po_nums]

def trigger_selection_hosts_selection(trigger_selection_mode,model,trigger_pool,host_samples,po_num,tr_num=1):
        _, trigger = Cer_triggers_selection(model, trigger_pool,tr_num)
        hosts_frommax,hosts_fromin = Inf_hosts_selection(model,trigger[0],host_samples,po_num)
        if trigger_selection_mode=='Cer':
            return trigger[0],hosts_frommax
        else:
            return trigger[0], hosts_fromin

def gen_trigger_variants_db(poison_num): #augmention adopted in lib_trigger_injection
    random.seed(10086)
    vatiants_db = [0, -5, -10, -15, -20, -25, -30, -35, -40]
    random_trigger_idx = random.sample(range(0, poison_num), poison_num)
    selection_vatiants_db = [vatiants_db[i % len(vatiants_db)] for i in random_trigger_idx]
    return selection_vatiants_db






































#