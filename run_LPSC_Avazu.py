"""
Train a Avazu dataset 
"""

import sys
import logging
import pandas
from deepctr_torch.models import *
from vertical_fl.fit_LPSC import TrainerLPSC
from utils.utils import configuration

# 1. Active party local training. 
def train_local_model(config): 
    config['local_features'] = [] # Paste randomly sampled local features here.
    config['boost_features'] = [ [], [], [], [], []] # Paste randomly sampled passive party boosting features here.
    config["train_local_flag"] = True
    config["save_model"] = True
    config["local_model"] = None
    configuration(config)

    logging.info("Initializing models ...")
    trainer = TrainerVFXOR(config=config)
    trainer.load_data()
    trainer.init_local_model()
    trainer.fit_local_model()
    trainer.post_processing()


# 2. Data pre-processing to generate D_p (residuals).
def test_local_model_and_save_to_csv(config):
    configuration(config)
    logging.info("Initializing models ...")
    config['record_data_flag'] = True
    config['local_features'] = []
    config['boost_features'] = []  # place holder
    config['local_model'] = "checkpoints/"  

    trainer = TrainerVFXOR(config=config)
    trainer.load_data()
    trainer.init_local_model()
    trainer.predict_local_model_and_save_to_csv()


# 3. VFL. 
def train_LPSC(config):
    configuration(config)
    config['local_features'] = []
    config['boost_features'] = []

    logging.info("Initializing models ...")
    trainer = TrainerVFXOR(config=config)
    trainer.init_all_models()
    trainer.load_data()
    trainer.fit_boosting_models()
    trainer.post_processing()


def train_VFXOR(config):
    config['local_features'] = []
    config['boost_features'] = [
    ]
    configuration(config)

    logging.info("Initializing models ...")
    trainer = TrainerVFXOR(config=config)
    trainer.init_all_models()
    trainer.load_data()
    trainer.fit_xor_models()
    trainer.post_processing()


if __name__ == '__main__':

    args = sys.argv
    if len(args) > 1:
        task = args[1]
        if task == "debug":
            task = 'AvazuDebug'
    else:
        task = "Avazu"

    config_boosting = {
        "task_name": task,
        "beta_unlearn": 0.,  
        "model": DeepFMLogit,  
        "batch_size": 4096,
        "epochs": 5,
        "l2_reg_linear": 1e-3,
        "l2_reg_embedding": 1e-3,
        "l2_reg_dnn": 1e-2,
        "task": "binary",
        "common_data_ratio": 0.5,
        "alpha_boosting": 2.0, 
        "optimizer": "adam",
        "boost_model_lr": 1e-4,
        "local_model_lr": 1e-3,
        "baseline_model_lr": 1e-3,
        "lr_decay_gamma": 0.0,    # lr scheduler, ExpoentialLR  gamma
        "loss": "binary_crossentropy",
        "metrics": ["binary_crossentropy", "auc"],
        "num_byzantine": 0,  # Byzantine Attack
        "byzantine_attack": "no",   # {"gaussian":  gaussian attack,  "sign_flip": sign_flip attack, "same_value", "no": no attack}
        "defence_type": "no",   # {1. "no": average on all logits,  2. "oracle": oracle type,   3."median", 
                                    # 4. "trimmed_mean", 5. "noisy_median", 6. "diverseFL", 7. "weight", 8. "diverseFL_weight"}
        "attack_in_test_phase": False,
        "defend_in_test_phase": True,
        "verbose": 1,
        "initial_epoch" : 0,
        "eval_interval": 1000,
        "boost_model_loss_type": "boosting",  # "boosting" or "averaging"
        "baseline_model": False,  # Use baseline model to test performance without FedBoosting, only local train.
        "share_embedding": False,   # True is problematic. 
        "log_into_file": True,
        "show_progress": True,
        "train_local_flag": False,
        "save_model": False, 
        "local_model": "checkpoints/pined/Avazu10M_C1-8_AUC6972.pt", 
        "local_model_AUC": 0.6972,
        "boost_model": None,
        "remove_saved_model_after_training": False,
        "ldp_rr": 0,   
        "record_data_flag": False,  # If flase, do not record sample-wise prediction results to CSV file for subsequential processing. 
        "feature_split_mode": 1,
        "xor_threshold": None,
        "xor_confidence": None, 
        "comment": "",
    }

    # train_local_model(config_boosting)
    # train_LPSC(config_boosting)
