"""
Train LPSC on MIMIC-III dataset
"""

import sys
import logging
import random
import pandas
from deepctr_torch.models import *
from vertical_fl.fit_LPSC import TrainerLPSC
from utils.utils import configuration

# 1. Active party local training. 
def train_local_model(config):
    config['local_features'] = [] # Paste randomly sampled local features here.
    config['boost_features'] = [ [], [], [], [], []] # Paste randomly sampled passive party boosting features here.
    config["train_local_flag"] = True
    config["save_model"] = False
    config["local_model"] = None
    configuration(config)

    logging.info("Initializing models ...")
    trainer = TrainerVFXOR(config=config)
    trainer.load_data()
    trainer.init_local_model()
    trainer.fit_local_model()

# 2. Data pre-processing to generate D_p (residuals).
def test_local_model_and_save_to_csv(config):
    configuration(config)
    logging.info("Initializing models ...")
    config['record_data_flag'] = True
    config['local_features'] =[]
    config['boost_features'] = []  
    config['local_model'] = "checkpoints/"  

    trainer = TrainerVFXOR(config=config)
    trainer.load_data()
    trainer.init_local_model()
    trainer.predict_local_model_and_save_to_csv()


# 3. VFL. 
def train_LPSC(config):
    configuration(config)
    config['local_features'] = []
    config['boost_features'] = random_features(num_passive_parties=6)
    config['epochs'] = 200
    logging.info("Initializing models ...")
    trainer = TrainerVFXOR(config=config)
    trainer.init_all_models()
    trainer.load_data()
    if config['boost_model_loss_type'] == "boosting":
        trainer.fit_boosting_models()
    else:
        config["alpha_local"] = 1./7
        config["alpha_boosting"] = 6./7
        trainer.fit_averaging_models()


def train_VFXOR(config):
    config['local_features'] = ["C1", "C2", "C3", "C4", "C5", "C6", 'C7', 'C8']
    config['boost_features'] = [
        ['C9', 'C10', 'C11'],   # 55
        ['C12', 'C23', 'C17'],  # 47
        ['C15', 'C16', 'C14'],  # 65
        ['C18', 'C19', 'C20'],  # 57
        ['C21', 'C22', 'C13']   # 64
    ]
    configuration(config)

    logging.info("Initializing models ...")
    trainer = TrainerVFXOR(config=config)
    trainer.init_all_models()
    trainer.load_data()
    trainer.fit_xor_models()


def random_features(num_passive_parties):
    # First, create a list of all feature indexes
    all_feature_indexes = list(range(100, 715))

    # Shuffle the feature indexes
    random.shuffle(all_feature_indexes)

    # Split the shuffled feature indexes into 6 parts
    split_sizes = [len(all_feature_indexes) // num_passive_parties for _ in range(num_passive_parties)]
    split_sizes[-1] += len(all_feature_indexes) % num_passive_parties

    shuffled_boost_features = []
    start = 0
    for size in split_sizes:
        shuffled_boost_features.append(["I{}".format(x) for x in all_feature_indexes[start:start+size]])
        start += size
    return shuffled_boost_features

if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        task = args[1]
        if task == "debug":
            task = 'MimicDebug'
    else:
        task = "MimicAll"

    config_boosting = {
        "task_name": task,
        "beta_unlearn": 0.05,     
        "model": MLPLogit,   
        "batch_size": 16911,
        "epochs": 200,
        "l2_reg_linear": 1e-3,
        "l2_reg_embedding": 1e-3, 
        "l2_reg_dnn": 1e-3,
        "task": "binary",
        "common_data_ratio": 1,
        "alpha_local": 1.0,  
        "alpha_boosting": 2.0,  
        "optimizer": "adam",
        "boost_model_lr": 1e-2,
        "local_model_lr": 1e-3,
        "baseline_model_lr": 1e-3,
        "lr_decay_gamma": 0.999,    # lr scheduler, ExpoentialLR  gamma
        "loss": "binary_crossentropy",
        "metrics": ["binary_crossentropy", "auc"],
        "num_byzantine": 0,  # Byzantine Attack
        "byzantine_attack": "no",   # {"gaussian":  gaussian attack,  "sign_flip": sign_flip attack, "same_value", "no": no attack}
        "defence_type": "no",   # {1. "no": average on all logits,  2. "oracle": oracle type,   3."median", 
                                    # 4. "trimmed_mean", 5. "noisy_median", 6. "diverseFL", 7. "weight", 8. "diverseFL_weight"}
        "attack_in_test_phase": False,
        "defend_in_test_phase": True,
        "verbose": 0,
        "initial_epoch" : 0,
        "eval_interval": 10,
        "boost_model_loss_type": "averaging",  # "boosting" or "averaging"
        "baseline_model": False,  # Use baseline model to test performance without FedBoosting, only local train.
        "share_embedding": False,   # True is problematic. 
        "log_into_file": True,
        "show_progress": True,
        "train_local_flag": False,
        "save_model": False, 
        "local_model": "checkpoints/pined/MimicAll_local_I100_AUC636.pt", 
        "local_model_AUC": 0.636,
        "boost_model": None,
        "remove_saved_model_after_training": False,
        "ldp_rr": 0.3,   # 0 : no LDP
        "record_data_flag": False,  # If flase, do not record sample-wise prediction results to CSV file for subsequential processing. 
        "feature_split_mode": 1,
        # "local_model_prediction_file": None,
        "xor_threshold": None,
        "xor_confidence": None, 
        "comment": "",
    }

    # train_local_model(config_boosting)

    # train_LPSC(config_boosting)
