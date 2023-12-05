import sys
import logging
import pandas
from deepctr_torch.models import *
from vertical_fl.fit_LPSC import TrainerLPSC
from utils.utils import configuration


def train(config):
    
    configuration(config)
    config['local_features'] = [] # Paste randomly sampled feature names here. 
    config['boost_features'] = [ [], [], [], [], [], []]  # Paste randomly sampled passive party features here.

    logging.info("Initializing models ...")

    # Train the model
    trainer = TrainerLPSC(config=config)
    trainer.init_all_models()
    trainer.load_data()
    trainer.fit()
    trainer.post_processing()



if __name__ == '__main__':

    args = sys.argv
    if len(args) > 1:
        task = args[1]
        if task == "debug":
            task = 'CriteoDebug'
    else:
        task = "Criteo10M"

    config_boosting = {
        "task_name": task,
        "beta_unlearn": 0.,
        "model": DeepFMLogit, #WDLLogit, #DeepFMLogit,
        "batch_size": 1024,
        "epochs": 1,
        "l2_reg_linear": 1e-4,
        "l2_reg_embedding": 1e-3,
        "task": "binary",
        "common_data_ratio": 0.5,
        "alpha_boosting": 2.0,  # [0, 0.5, 1.0, 2.0, 3.0, 4.0 ],
        "optimizer": "adam",
        "boost_model_lr": 1e-3,
        "local_model_lr": 1e-3,
        "baseline_model_lr": 1e-3,
        "lr_decay_gamma": 0.0,    # lr scheduler, ExpoentialLR  gamma
        "loss": "binary_crossentropy",
        "metrics": ["binary_crossentropy", "auc"],
        "alpha_kd": 0.7,   # [0.2, 0.5, 0.7]
        "temperature": 10,     # [3, 10, 20]  T = 10 win
        "knowledge_distillation_flag": True,
        "num_byzantine": 0,  # Byzantine Attack
        "byzantine_attack": "no",   # {"gaussian":  gaussian attack,  "sign_flip": sign_flip attack, "same_value", "no": no attack}
        "defence_type": "weight",   # {1. "no": average on all logits,  2. "oracle": oracle type,   3."median", 
                                    # 4. "trimmed_mean", 5. "noisy_median", 6. "diverseFL", 7. "weight", 8. "diverseFL_weight"}
        "attack_in_test_phase": False,
        "defend_in_test_phase": True,
        "verbose": 1,
        "initial_epoch" : 0,
        "eval_interval": 1000,
        "boost_model_loss_type": "averaging",  # "boosting" or "averaging"
        "baseline_model": False,  # Use baseline model to test performance without FedBoosting, only local train.
        "share_embedding": False,   # True is problematic. 
        "log_into_file": True,
        "show_progress": True,
        "feature_split_mode": 2,
        # "save_model": True,   # Train new local model 
        # "local_model": None, # "checkpoints/Criteo10M_local_Oct-31-2022-17h-59m-24s.pt",
        # "train_local_flag": True,
        "save_model": False, 
        "local_model": "checkpoints/Criteo10M_local_Dec-05-2022-14h-16m-45s.pt",   # Split mode 2
        "train_local_flag": False,
        "boost_model": None,
        "remove_saved_model_after_training": False,
        "embedding_vocabulary": 'embedding_dicts/vocabulary_size_10M.json',
        "ldp_rr": 0,   # 0 : no LDP
        "comment": "",
    }
    train(config_boosting)


    # "local_model":  "checkpoints/Criteo10M_local_Oct-31-2022-17h-59m-24s.pt",  # Split mode 1
