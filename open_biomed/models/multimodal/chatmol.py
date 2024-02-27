import logging
logger = logging.getLogger(__name__)

import os
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

from open_biomed.models.base_models import MolEncoder, TextEncoder

class ChatMol(MolEncoder, TextEncoder):
    def __init__(self, config):
        super(ChatMol, self).__init__()
        t5_config = T5Config.from_json_file(os.path.join(config["model_name_or_path"], "config.json"))
        self.main_model = T5ForConditionalGeneration(t5_config)
        if "ckpt" in config:
            logger.info("Load checkpoint from %s" % (config["ckpt"]))
            state_dict = torch.load(config["ckpt"], map_location="cpu").state_dict()
            self.main_model.load_state_dict(state_dict)
        self.decoder_tokenizer = T5Tokenizer.from_pretrained(config["model_name_or_path"])

    def forward(self, encoder_outputs, attention_mask, decoder_attention_mask, labels):
        return self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        ).loss

    def encode_mol(self, mol):
        return self.main_model.encoder(**mol).last_hidden_state

    def encode_text(self, text):
        return self.main_model.encoder(**text).last_hidden_state

    def decode(self, encoder_outputs, attention_mask, num_beams, max_length):
        outputs = self.main_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length
        )
        return outputs
        #return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)