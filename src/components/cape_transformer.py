#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb as pdb
from transformers import TransfoXLModel, TransfoXLConfig
from src.components.self_attention import MultiHeadedAttention
from src.components.transformer_encoder import Encoder, EncoderLayer, EncoderLayerFFN
from src.components.position_encodings import 	PositionalEncoding, CosineNpiPositionalEncoding, LearnablePositionalEncoding
from src.components.cape_new import CAPE1d
from torch import nn

class CAPETransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken,noutputs, d_model, nhead, d_ffn, nlayers, dropout=0.5, use_embedding=False, pos_encode = True, bias = False, pos_encode_type = 'absolute', max_period = 10000.0):
        super(CAPETransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None

        # if use_embedding:
        # 	self.pos_encoder = PositionalEncoding(d_model, dropout)
        # 	self.encoder = nn.Embedding(ntoken, d_model)
        # else:
        # 	self.pos_encoder = PositionalEncoding(ntoken, dropout)
        # 	self.encoder = nn.Embedding(ntoken, ntoken)
        # 	self.encoder.weight.data =torch.eye(ntoken)
        # 	self.encoder.weight.requires_grad = False

        self.pos_encoder = CAPE1d(d_model, max_global_shift=5.0, max_local_shift=0.5,max_global_scaling=1.03)
        self.encoder = nn.Embedding(ntoken, d_model)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_ffn, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.d_model = d_model
        self.decoder = nn.Linear(d_model, noutputs, bias=bias)
        self.sigmoid= nn.Sigmoid()
        self.bias = bias

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.bias:
            self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True, get_attns = False, get_encoder_reps = False):
        if has_mask:
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        else:
            self.src_mask = None
        src = self.encoder(src) * math.sqrt(self.d_model)
       
        src = self.pos_encoder(src)
       
        if get_attns:
            attns = []
            encoder_layers = self.transformer_encoder.layers
            inp = src
            for layer in encoder_layers:
                attn = layer.self_attn(inp, inp, inp, attn_mask = self.src_mask)[1]
                inp = layer(inp, src_mask = self.src_mask) 
                attns.append(attn)


        transformer_output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(transformer_output)
        output = self.sigmoid(output)
        # return F.log_softmax(output, dim=-1)

        if get_attns:
            return output, attns	

        if get_encoder_reps:
            return output, transformer_output

        return output

