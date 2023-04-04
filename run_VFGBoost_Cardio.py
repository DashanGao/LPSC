"""
Train on Cardio dataset.
"""

import sys
import logging
import random
import pandas
from deepctr_torch.models import *
from vertical_fl.fit_VFGBoost import TrainerVFGBoost
from utils.utils import configuration

# 1. Active party local training. 
def train_local_model(config):
    feat = random_features(7, 1, 246)
    config['local_features'] = feat[0] 
    config['boost_features'] = feat[1:]
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
    config['local_features'] =["I{}".format(x) for x in range(1, 100)]
    config['boost_features'] = []  # place holder
    config['local_model'] = "checkpoints/Cardi_local_I100_AUC.pt"  

    trainer = TrainerVFXOR(config=config)
    trainer.load_data()
    trainer.init_local_model()
    trainer.predict_local_model_and_save_to_csv()


# 3. VFL. 
def train_VFGBoost(config):
    configuration(config)
    config['local_features'] = [] # Randomly samples local features
    config['boost_features'] = random_features_exclude(num_passive_parties=6, start=1, end=246, to_exclude=config['local_features'])

    config['epochs'] = 100
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


def random_features(num_passive_parties, start, end):
    # First, create a list of all feature indexes
    all_feature_indexes = list(range(start, end))

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


def random_features_exclude(num_passive_parties, start, end, to_exclude):
    # Create a list of all feature indexes
    all_feature_indexes = list(range(start, end + 1))
    
    # Remove the feature indexes specified in the 'to_exclude' list
    excluded_indexes = [int(s[1:]) for s in to_exclude]
    all_feature_indexes = [x for x in all_feature_indexes if x not in excluded_indexes]

    # Shuffle the remaining feature indexes
    random.shuffle(all_feature_indexes)

    # Split the shuffled feature indexes into 'num_passive_parties' parts
    split_sizes = [len(all_feature_indexes) // num_passive_parties for _ in range(num_passive_parties)]
    split_sizes[-1] += len(all_feature_indexes) % num_passive_parties

    shuffled_features = []
    start = 0
    for size in split_sizes:
        shuffled_features.append(["I{}".format(x) for x in all_feature_indexes[start:start+size]])
        start += size
    return shuffled_features

if __name__ == '__main__':
    config_boosting = {
        "task_name": 'Cardi',
        "beta_unlearn": 0.,     # Adversarial training for Trade-Off
        "model": MLPLogit,   # DeepFMLogit, #WDLLogit, #DeepFMLogit,
        "batch_size": 4000,
        "epochs": 400,
        "l2_reg_linear": 1e-3,
        "l2_reg_embedding": 0, #1e-3,
        "l2_reg_dnn": 1e-3,
        "task": "binary",
        "common_data_ratio": 1,
        "alpha_local": 1.0,  
        "alpha_boosting": 1.0, 
        "optimizer": "adam",
        "boost_model_lr": 1e-3,
        "local_model_lr": 1e-3,
        "baseline_model_lr": 1e-3,
        "lr_decay_gamma": 0.999,    # lr scheduler, ExpoentialLR  gamma
        "loss": "binary_crossentropy",
        "metrics": ["binary_crossentropy", "auc"],
        "num_byzantine": 0,  # Byzantine Attack
        "byzantine_attack": "no",   # {"gaussian":  gaussian attack,  "sign_flip": sign_flip attack, "same_value", "no": no attack}
        "defence_type": "weight",   # {1. "no": average on all logits,  2. "oracle": oracle type,   3."median", 
                                    # 4. "trimmed_mean", 5. "noisy_median", 6. "diverseFL", 7. "weight", 8. "diverseFL_weight"}
                                # FIXME: when 2PC, should use "no".
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
        "local_model": "checkpoints/pined/Cardi_local_I99r_AUC6636.pt", 
        "local_model_AUC": 0.6636,
        "boost_model": None,
        "remove_saved_model_after_training": False,
        "ldp_rr": 0.5,   # 0 : no LDP
        "dp_Lap_epsilon": .8,
        "record_data_flag": False,  
        "feature_split_mode": 1,
        "xor_threshold": None,
        "xor_confidence": None, 
        "comment": "",
    }

    # train_local_model(config_boosting)

    # train_VFGBoost(config_boosting)
    
