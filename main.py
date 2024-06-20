import os
import hydra
import logging
import datetime
import tensorflow as tf
from omegaconf import DictConfig

from src.trainer.trainer import Trainer
from src.utils.utils import check_dir
from src.utils.utils import logger, logging_mode
from src.utils.utils import TruncatePathFormatter


def train(cfg: DictConfig):
    runner = Trainer(cfg)
    runner.train()


def test(cfg: DictConfig):
    runner = Trainer(cfg)
    runner.test()


@hydra.main(version_base=None, config_path="config", config_name="base_config.yaml")
def main(cfg: DictConfig):
    # Log setting
    logging_configure(cfg=cfg.general.logging)

    # Use GPU or not
    if not cfg.general.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        physical_devices = tf.config.list_physical_devices('GPU')
        # try:
        #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # except:
        #     exit("GPU allocated failed")

    # Train or Test
    if cfg.general.mode == 'train':
        train(cfg)
    elif cfg.general.mode == 'test':
        if cfg.general.checkpoint is None:
            exit("Please load the pretrained checkpoint")
        else:
            test(cfg)
    else:
        raise RuntimeError('No such a mode, please check it')


def logging_configure(cfg: DictConfig):
    # Remove all handlers associated with the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Logging settings
    log_mode = logging_mode(cfg.mode)
    logger.setLevel(log_mode)
    # formatter = TruncatePathFormatter('[%(asctime)s] [%(pathname)s] [%(levelname)s]: %(message)s', datefmt='%H:%M:%S')
    formatter = TruncatePathFormatter('[%(asctime)s] [%(filename)s] [%(levelname)s]: %(message)s', datefmt='%H:%M:%S')

    # Logging into file
    if cfg.mode is not None and cfg.folder is not None:
        check_dir(cfg.folder)
        filename = '{}.log'.format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        pathname = os.path.join(cfg.folder, filename)

        file_handler = logging.FileHandler(pathname)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_mode)
        logger.addHandler(file_handler)

    # Logging into console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_mode)
    logger.addHandler(console_handler)


if __name__ == '__main__':
    main()
