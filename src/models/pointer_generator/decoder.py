import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import TransformerDecoderLayer
from torch.nn.modules.activation import MultiheadAttention
from torch.nn import Linear, Dropout
from torch.nn import LayerNorm
from torch.nn.modules.transformer import _get_activation_fn, _get_clones


class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers
    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        decoder_final_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    Examples::
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        memory = torch.rand(10, 32, 512)
        tgt = torch.rand(20, 32, 512)
        out = transformer_decoder(tgt, memory)
    """

    def __init__(self, decoder_layer, decoder_final_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers - 1)
        self.final_layer = decoder_final_layer
        self.num_layers = num_layers
        self.norm = norm
        self.linear()

    def forward(self, tgt, enc_outputs, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        Args:
            tgt: the sequence to the decoder (required).
            enc_outputs: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        # output = enc_outputs[:, :, :1]
        output = enc_outputs[:,:11,:].transpose(0,1).cuda()

        enc_outputs = enc_outputs.transpose(0, 1).cuda()
        # output = output.transpose(0, 1).cuda()
        tgt_mask = tgt_mask.cuda()

        # Run through "normal" decoder layer
        for i in range(self.num_layers - 1):
            # output = self.layers[i](output, enc_outputs, tgt_mask=tgt_mask.cuda(), memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask.bool().cuda(), memory_key_padding_mask=memory_key_padding_mask.bool().cuda())
            output = self.layers[i](output, enc_outputs)
        # Run through final decoder layer, which outputs the attention weights as well
        # output, attention_weights = self.final_layer(output, enc_outputs, tgt_mask=tgt_mask.cuda(), memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask.bool().cuda(), memory_key_padding_mask=memory_key_padding_mask.bool().cuda())
        import pdb;pdb.set_trace()

        output, attention_weights = self.final_layer(output, enc_outputs)
        if self.norm:
            output = self.norm(output)

        return output, attention_weights

class TransformerDecoderFinalLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
        Layer also output attention weights from the multi-head-attn, used for pointer-generator model.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        memory = torch.rand(10, 32, 512)
        tgt = torch.rand(20, 32, 512)
        out, attention = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderFinalLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # Model saves attention weights from multi-head-attn
        tgt2, attention_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # for backward compatibility
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attention_weights