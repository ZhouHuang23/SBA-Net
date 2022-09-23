import os
from config.config import cfg
import torch
from core.libs.logger import set_logger
from core.inference.infer_SSBANet_vgg16 import infer_TestNet

logger = set_logger()


class Inference:
    def __init__(self, well_trained_model):
        """
        @param well_trained_model: trained model
        """
        self.model = well_trained_model
        weight_path = os.path.join(cfg.CKPT.SAVE_DIR, cfg.CKPT.SELECTED_INFER_CKPT)

        self.model.load_state_dict(torch.load(weight_path)['net'])
        self.model.eval()
        logger.info('Model {} checkpoint weight has been loaded successfully...'.format(cfg.MODEL.NAME))

    def run(self):
        logger.warning(">>>> Start inference using general prediction manner .")
        infer_TestNet(self.model)
