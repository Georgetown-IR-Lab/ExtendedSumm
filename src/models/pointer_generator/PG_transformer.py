import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from models.pointer_generator.utils import PositionalEncoding, _generate_square_subsequent_mask, Embedding
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer
from models.pointer_generator.decoder import TransformerDecoder, TransformerDecoderFinalLayer


class PointerGeneratorTransformer(nn.Module):
    def __init__(self, src_vocab_size=128, tgt_vocab_size=128,
                 embedding_dim=768, fcn_hidden_dim=128,
                 num_heads=4, num_layers=2, dropout=0.2,
                 src_to_tgt_vocab_conversion_matrix=None):
        super(PointerGeneratorTransformer, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.src_to_tgt_vocab_conversion_matrix = src_to_tgt_vocab_conversion_matrix
        self.pos_encoder = PositionalEncoding(embedding_dim)
        # Source and target embeddings
        self.src_embed = Embedding(self.src_vocab_size, embedding_dim, padding_idx=2)
        self.tgt_embed = Embedding(self.tgt_vocab_size, embedding_dim, padding_idx=2)

        # Decoder layers
        self.decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, fcn_hidden_dim, dropout)
        self.decoder_final_layer = TransformerDecoderFinalLayer(embedding_dim, num_heads, fcn_hidden_dim, dropout)
        self.decoder = TransformerDecoder(self.decoder_layer, self.decoder_final_layer, num_layers)

        # Final linear layer + softmax. for probability over target vocabulary

        # Context vector
        self.c_t = None

        # Initialize masks
        self.src_mask = None
        self.tgt_mask = None
        self.mem_mask = None
        # Initialize weights of model
        self._reset_parameters()

    def _reset_parameters(self):
        """ Initiate parameters in the transformer model. """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def decode(self, enc_outputs, tgt, src, tgt_key_padding_mask=None, memory_key_padding_mask=None, has_mask=True):
        """
        Applies embedding, positional encoding on target  and then runs the transformer encoder on the memory and target.
        Also creates square subsequent mask for teacher learning.
        :param enc_outputs: The encoder hidden states
        :param tgt: Target tokens batch
        :param tgt_key_padding_mask: target padding mask
        :param memory_key_padding_mask: memory padding mask
        :param has_mask: Whether to use square subsequent mask for teacher learning
        :return: decoder output
        """

        # Create target mask for transformer if no appropriate one was created yet, created of size (T, T)
        if has_mask:
            if self.tgt_mask is None or self.tgt_mask.size(0) != tgt.size(1):
                self.tgt_mask = _generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        else:
            self.tgt_mask = None

        # Target embedding and positional encoding, changes dimension (N, T) -> (N, T, E) -> (T, N, E)
        # tgt_embed = self.tgt_embed(tgt).transpose(0, 1)
        tgt_embed_pos = self.pos_encoder(tgt)
        # Get output of decoder and attention weights. decoder Dimensions stay the same

        # tgt_key_padding_mask = torch.ones(size=(tgt.size(0), tgt.size(1)))
        # memory_key_padding_mask = torch.ones(size=(enc_outputs.size(0), enc_outputs.size(1)))
        import pdb;pdb.set_trace()
        decoder_output, attention = self.decoder(tgt_embed_pos, enc_outputs, tgt_mask=self.tgt_mask, memory_mask=self.mem_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        # Get probability over target vocabulary, (T, N, E) -> (T, N, tgt_vocab_size)
        p_vocab = self.p_vocab(decoder_output)

        # ---Compute Pointer Generator probability---
        # Get hidden states of source (easier/more understandable computation). (S, N, E) -> (N, S, E)
        hidden_states = enc_outputs.transpose(0, 1)
        # compute context vectors. (N, T, S) x (N, S, E) -> (N, T, E)
        context_vectors = torch.matmul(attention, hidden_states).transpose(0, 1)
        total_states = torch.cat((context_vectors, decoder_output, tgt_embed), dim=-1)
        # Get probability of generating output. (N, T, 3*E) -> (N, T, 1)
        p_gen = self.p_gen(total_states)
        # Get probability of copying from input. (N, T, 1)
        p_copy = 1 - p_gen

        # Get representation of src tokens as one hot encoding
        one_hot = torch.zeros(src.size(0), src.size(1), self.src_vocab_size, device=src.device)
        one_hot = one_hot.scatter_(dim=-1, index=src.unsqueeze(-1), value=1)
        # p_copy from source is sum over all attention weights for each token in source
        p_copy_src_vocab = torch.matmul(attention, one_hot)
        # convert representation of token from src vocab to tgt vocab
        p_copy_tgt_vocab = torch.matmul(p_copy_src_vocab, self.src_to_tgt_vocab_conversion_matrix).transpose(0,
                                                                                                                  1)
        # Compute final probability
        p = torch.add(p_vocab * p_gen, p_copy_tgt_vocab * p_copy)

        # Change back batch and sequence dimensions, from (T, N, tgt_vocab_size) -> (N, T, tgt_vocab_size)
        return torch.log(p.transpose(0, 1))

    # def forward(self, src, tgt, enc_outputs, src_key_padding_mask=None, tgt_key_padding_mask=None,
    #             memory_key_padding_mask=None, has_mask=True):
    def forward(self, enc_outputs, src, tgt):
        """Take in and process masked source/target sequences.
		Args:
			src: the sequence to the encoder (required).
			tgt: the sequence to the decoder (required).
			src_mask: the additive mask for the src sequence (optional).
			tgt_mask: the additive mask for the tgt sequence (optional).
			memory_mask: the additive mask for the encoder output (optional).
			src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).
		Shape:
			- src: :math:`(S, N, E)`. Starts as (N, S) and changed after embedding
			- tgt: :math:`(T, N, E)`. Starts as (N, T) and changed after embedding
			- src_mask: :math:`(S, S)`.
			- tgt_mask: :math:`(T, T)`.
			- memory_mask: :math:`(T, S)`.
			- src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.
			Note: [src/tgt/memory]_mask should be filled with
			float('-inf') for the masked positions and float(0.0) else. These masks
			ensure that predictions for position i depend only on the unmasked positions
			j and are applied identically for each sequence in a batch.
			[src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
			that should be masked with float('-inf') and False values will be unchanged.
			This mask ensures that no information will be taken from position i if
			it is masked, and has a separate mask for each sequence in a batch.
			- output: :math:`(T, N, E)`.
			Note: Due to the multi-head attention architecture in the transformer model,
			the output sequence length of a transformer is same as the input sequence
			(i.e. target) length of the decode.
			where S is the source sequence length, T is the target sequence length, N is the
			batch size, E is the feature number
		Examples:
			output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
		"""

        # Applies embedding, positional encoding and the transformer encoder on the source
        # Applies embedding, positional encoding on target  and then runs the transformer encoder on the memory and target.

        """
            tgt should be sentence embeddings for target
        """

        tgt_key_padding_mask = torch.ones(size=(tgt.size(0), tgt.size(1)))
        memory_key_padding_mask = torch.ones(size=(enc_outputs.size(0), enc_outputs.size(1)))
        has_mask = True
        output = self.decode(enc_outputs, tgt, src, tgt_key_padding_mask, memory_key_padding_mask, has_mask)
        # output = self.decode(enc_outputs, tgt, src)
        return output