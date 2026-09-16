import sys
import torch
import torch.nn as nn

sys.path.append("../")
from layers.models_mae import *

class CustomVLM(nn.Module):
    def __init__(self, config, device):
        super(CustomVLM, self).__init__()
        self.config = config
        self.device = device
        
        self.hidden_size = 768
        
        self._init_vision_encoder()
        self._init_text_encoder()

    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _init_vision_encoder(self):
        from transformers import ViTModel, ViTFeatureExtractor
        self.vision_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vision_encoder.to(self.device)
        self._set_requires_grad(self.vision_encoder, self.config.finetune_vlm)

    def _init_text_encoder(self):
        from transformers import BertTokenizer, BertModel
        self.text_processor = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_encoder.to(self.device)
        self._set_requires_grad(self.text_encoder, self.config.finetune_vlm)

    def _set_requires_grad(self, model, value):
        for param in model.parameters():
            param.requires_grad = value

    def get_vision_embeddings(self, images):
        inputs = self.vision_processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.vision_encoder(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def get_text_embeddings(self, texts):
        inputs = self.text_processor(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]
