import logging
logger = logging.getLogger(__name__)

import os
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration

from open_biomed.models.base_models import MolEncoder, ProteinEncoder, TextEncoder
from open_biomed.utils.mol_utils import get_biot5_tokenizer
from torch_geometric.nn import (MessagePassing, global_add_pool, global_max_pool, global_mean_pool)

activation = {
    "sigmoid": nn.Sigmoid(),
    "softplus": nn.Softplus(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
}

class MLP(nn.Module):
    def __init__(self, config, input_dim, output_dim=1):
        super(MLP, self).__init__()
        self.model = nn.Sequential()
        hidden_dims = [input_dim] + config["hidden_size"] + [output_dim]
        for i in range(len(hidden_dims) - 1):
            self.model.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i != len(hidden_dims) - 2:
                self.model.append(nn.Dropout(config["dropout"]))
                if config["activation"] != "none":
                    self.model.append(activation[config["activation"]])
                if config["batch_norm"]:
                    self.model.append(nn.BatchNorm1d())
    
    def forward(self, h):
        return self.model(h)


class BioT5(MolEncoder, ProteinEncoder, TextEncoder):
    def __init__(self, config, task_num=1):
        super(BioT5, self).__init__()
        t5_config = T5Config.from_json_file(os.path.join(config["model_name_or_path"], "config.json"))
        self.main_model = T5ForConditionalGeneration(t5_config)
        self.decoder_tokenizer = get_biot5_tokenizer(config)
        self.main_model.resize_token_embeddings(len(self.decoder_tokenizer))

        if "ckpt" in config:
            logger.info("Load checkpoint from %s" % (config["ckpt"]))
            state_dict = torch.load(config["ckpt"], map_location="cpu")
            self.main_model.load_state_dict(state_dict, strict=True)

        self.pred_head = MLP(config["pred_head"], 768, task_num)

    def forward(self, encoder_outputs, attention_mask, decoder_attention_mask, labels):
        return self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        ).loss

    # TODO: like mol caption config selffish [:,0,:] for pred_head
    def encode_mol(self, mol):
        return self.main_model.encoder(**mol).last_hidden_state

    def encode_protein(self, prot):
        return self.main_model.encoder(**prot).last_hidden_state

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

    def get_dp_output(self, mol):
        mol_embeds = self.encode_mol(mol)  
        # just use[:,0,:]
        h = mol_embeds[:, 0, :]
        return self.pred_head(h)