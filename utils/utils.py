import os
import logging
import logging.config
import json, copy
from datetime import date
from time import strftime, localtime, time

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter


LOCAL = 0
GOOLE_DRIVE = 1
INSPUR = 2
PC = 3


def init_tensorboard(config):
    # log_dir = '{}_{}_{}_bat{}_cdr{}_aKD{}_aBst{}_T{}'.format(config['task_name'], str(config['model']), 
    #             config['boost_model_loss_type'], config['batch_size'], 
    #             config['common_data_ratio'], config['alpha_kd'], config['alpha_boosting'], config['temperature'])
    # log_dir = log_dir.split("<class")[0] + log_dir.split("<class")[1].split("'>")[0].split(".")[
    #     -1] + log_dir.split("'>")[-1]
    
    # log_dir = '{}_{}_{}_bzt{}_def{}_cdr{}_aKD{}_aBst{}_T{}_{}_{}'.format(config['task_name'], str(config['num_byzantine']), 
    #             config['boost_model_loss_type'], config['byzantine_attack'], config['defence_type'],
    #             config['common_data_ratio'], config['alpha_kd'], 
    #             config['alpha_boosting'], config['temperature'], 
    #             config["figureprint"], config['comment'])
    
    # log_dir = '{}_{}_{}_{}_beta{}_{}_fsm{}_{}'.format(
    #             config['task_name'], 
    #             str(config['num_byzantine']), 
    #             config['byzantine_attack'], 
    #             config['defence_type'], 
    #             config['beta_unlearn'], 
    #             config['comment'], 
    #             config['figureprint'], 
    #             config['feature_split_mode'])

    log_dir = '{}_{}_{}'.format(
                config['task_name'],
                config['comment'], 
                config['figureprint'])

    tensorboard = SummaryWriter(log_dir="tensorboard/" + log_dir)
    config_tmp = copy.copy(config)
    config_tmp["model"] = str(config_tmp["model"]).split("<class")[1].split("'>")[0].split(".")[-1]
    tensorboard.add_text(tag='argument', text_string=json.dumps(config_tmp, indent=6))
    return tensorboard


def set_logger(log_path=None):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        if log_path:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps)


def check_env():
    cwd = os.getcwd()
    if cwd[:10] == "/Users/gds":
        env = LOCAL
    elif cwd[:10] == "/content/d":
        env = GOOLE_DRIVE
    elif cwd[:10] == "/home/gds/":
        env = INSPUR
    elif cwd[:10] == "/home/dash":
        env = PC
    else:
        print("ERROR: Code location judge wrong!")
        exit()
    return env


ENV_LOC = check_env()


def get_projcet_path():
    if ENV_LOC == LOCAL:
        path = "/Users/gds/Documents/Research/open_source_libs/ctr_prediction/DeepCTR-Torch/deepctr-torch/"
    elif ENV_LOC == GOOLE_DRIVE:
        path = "/content/drive/My Drive/research_experiments/deepctr-torch/"
    elif ENV_LOC == INSPUR:
        path = "/home/gds/code/"
    elif ENV_LOC == PC:
        path = "/home/dashan/Document/research/ctr_prediction/deepctr-torch/"
    else:
        path = os.getcwd()
        print("Project path: ", os.getcwd())
    return path



def read_file():
    with open(get_projcet_path() + "raw_data/ml-100k/hetro_split_dataset/movielens100k_full.csv", "r") as src:
        lines = src.readlines()
    print(len(lines))
    print(lines[1])


def save_training_result(hyper_parameters, markdown_table):
    """
    save for formated training results to two files.
    :param hyper_parameters: a dict of result
    :param markdown_table:
    :return:
    """
    hyper_parameters["model"] = str(hyper_parameters["model"])
    hyper_parameters_str = json.dumps(hyper_parameters, ensure_ascii=True, indent="\t") + "\n"
    with open("logging/hyper_parameters" + hyper_parameters["task_name"] + ".txt", "a+") as des:
        des.write(hyper_parameters_str)
    with open("logging/markdown_table" + hyper_parameters["task_name"] + ".md", "a+") as des:
        des.write(markdown_table + "\n")


def load_data(data_path, local_prediction_file=None):
    """
    Load data from file
    :param data_path:
    :return:
    """
    with open(data_path, "r") as src:
        head = src.read(1)
    column_name_flag = (head == "0" or head == "1")
    del head
    if column_name_flag:  # No column name in csv file
        sparse_features = ['C' + str(i) for i in range(1, 27)]
        dense_features = ['I' + str(i) for i in range(1, 14)]
        data_column_names = ['label'] + dense_features + sparse_features
        data = pd.read_csv(data_path, names=data_column_names)
    else:
        data = pd.read_csv(data_path, delimiter=",")
        sparse_features = [x for x in data.columns if x[0] == "C"]
        dense_features = [x for x in data.columns if x[0] == "I"]
    
    data[sparse_features].astype(int)
    data[dense_features].astype(float)
    data['label'].astype(int)
    logging.info("Data read successfully.")
    if local_prediction_file is not None and os.path.exists(local_prediction_file):
        local_prediction = pd.read_csv(local_prediction_file)
        assert local_prediction.shape[0] == data.shape[0], "local predition csv miss-match the data csv."
        local_prediction.reset_index(inplace=True, drop=True)
        data = pd.concat([data, local_prediction[['local_prediction', 'gt_residual']]], axis=1)
        data[['local_prediction','gt_residual']].astype(float)
        logging.info("Local prediction read successfully.")

    return data, sparse_features, dense_features


def train_test_split_data(data, train_data_path, test_data_path, test_size=0.2):
    data = data.sample(frac=1.0)
    train, test = train_test_split(data, test_size=test_size)
    try:
        train.to_csv(train_data_path)
        test.to_csv(test_data_path)
        logging.info("Train / test data splitted and saved with test_size={0}".format(test_size))
    except Exception as e:
        logging.error(str(e))
        exit()
    data = pd.concat([train, test], axis=0)
    return data, train.shape[0]


def get_best_performane_marker(model):
    best_performance_marker = 0
    try:
        if "auc" in model.metrics.keys() or "accuracy" in model.metrics.keys() or "acc" in model.metrics.keys():
            best_performance_marker = 0
        elif "mse" in model.metrics.keys() or "binary_crossentropy" in model.metrics.keys() or "logloss" in model.metrics.keys():
            best_performance_marker = 1e7
        if best_performance_marker == -1:
            raise NameError("The metric is not in {auc, accuracy, acc, nse, binary_crossentropy, logloss}.")
    except Exception as e:
        print("Error: ", e)
        exit()
    return best_performance_marker


def process_input(input, feature_list):
    """
    Read input data and process it for dataloader.
    :param input:
    :param feature_list: model.feature_index, OrderedDict():  { feat_name: {start_index, end_index} }
    :return:
    """
    if isinstance(input, dict) or isinstance(input, pd.DataFrame):
        # feature in feature_list: feat_name. feature_list is linear_feature_columns + dnn_feature_columns
        input = [input[feature_name] for feature_name in feature_list]
    # if isinstance(input, pd.DataFrame):
    #     input = [input[feature] for feature in feature_list]
    for i in range(len(input)):
        if len(input[i].shape) == 1:
            input[i] = np.expand_dims(input[i], axis=1)
    return np.concatenate(input, axis=-1)


def configuration(config):
    """Process configureation. 
    Initialize logger. 

    Args:
        config (dict): dict of congurtions. 

    Returns:
        dict: config
    """
    # Configuration: Set output file names
    PROJECT_PATH = get_projcet_path()

    task_name = config["task_name"]
    timestamp = date.today().strftime("%b-%d-%Y") + "-" + strftime("%Hh-%Mm-%Ss", localtime())
    log_model_file_name = task_name + "_" + timestamp
    config['figureprint'] = date.today().strftime("%b-%d") + "-" + strftime("%Hh-%Mm-%Ss", localtime())
    if "Criteo" in task_name:
        config['data_path'] = PROJECT_PATH + "raw_data/criteo/" + task_name + ".csv"
        config["embedding_vocabulary"] = PROJECT_PATH + 'raw_data/criteo/embedding_dicts/vocabulary_size_10M.json'
    elif "Avazu" in task_name:
        config['data_path'] = PROJECT_PATH + "raw_data/avazu/" + task_name + ".csv"
        config["embedding_vocabulary"] = PROJECT_PATH + 'raw_data/avazu/embedding_dicts/vocabulary_size_10M.json'
    elif "Mimic" in task_name:
        config['data_path'] = PROJECT_PATH + "raw_data/mimic/" + task_name + ".csv"
        config["embedding_vocabulary"] = None
    elif "Cardi" in task_name:
        config['data_path'] = PROJECT_PATH + "raw_data/cardi/" + task_name + ".csv"
        config["embedding_vocabulary"] = None
    elif "Spambase" in task_name:
        config['data_path'] = PROJECT_PATH + "raw_data/spambase/" + task_name + ".csv"
        config["embedding_vocabulary"] = None
    
    if config["local_model"] is not None:
        config["local_model_prediction_file"] = config['data_path'].replace(task_name + ".csv", config["local_model"].split('/')[-1].split('.')[0] + ".csv")
    else:
        config["local_model_prediction_file"] = None
    
    logger_file_path = "logging/" + task_name + "/" + log_model_file_name + ".log"
    if config["log_into_file"]:
        set_logger(logger_file_path)
    else:
        set_logger()

    if not os.path.exists(PROJECT_PATH + "logging/" + task_name):
        os.makedirs(PROJECT_PATH + "logging/" + task_name)
    local_model_path = "checkpoints/" + task_name + "_local_" + timestamp + ".pt"
    boost_model_path = "checkpoints/" + task_name + "_boost_" + timestamp + ".pt"

    config["logger_file_path"] = logger_file_path
    config["des_path"] = {"local_model_path": local_model_path, "boost_model_path": boost_model_path}
    log_hyper_param = "Model hyper-parameters: " + str(config) + "\n" \
                      + "\n".join([": ".join([str(x1), str(x2)]) for [x1, x2] in config.items()]) + "\n"
    logging.info(task_name + " program start")
    logging.info(log_hyper_param)
    return config


def split_features(features, num_parties):
    pass



if __name__ == '__main__':
    print(ENV_LOC)
    print(get_projcet_path())
    read_file()
