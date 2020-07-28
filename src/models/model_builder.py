import copy

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
# from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
from torch.utils import checkpoint

from models.decoder import TransformerDecoder, TransformerDecoderState
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer


def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    return optim


def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if (large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            # self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
            self.model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def custom_sent_decider(self, module):
        def custom_forward(*inputs):
            output,_ = module(inputs[0],
                            attention_mask=inputs[1],
                            token_type_ids=inputs[2],
                            )
            return output

        return custom_forward

    def forward(self, x, segs, mask):
        if (self.finetune):

            top_vec, _ = self.model(x, attention_mask=mask, token_type_ids=segs)

            # top_vec = checkpoint.checkpoint(
            #     self.custom_sent_decider(self.model),
            #     x, mask, segs,
            # )

        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, attention_mask=mask, token_type_ids=segs)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint, is_joint=False):
        super(ExtSummarizer, self).__init__()
        self.is_joint = is_joint

        self.sentence_encoder = SentenceEncoder(args, device, checkpoint)
        self.sentence_predictor = SentenceExtLayer()

        # for p in self.sentence_encoder.parameters():
        #     if p.dim() > 1:
        #         xavier_uniform_(p)

        if is_joint:
            self.section_predictor = SectionExtLayer()

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for p in self.sentence_predictor.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            if self.is_joint:
                for p in self.section_predictor.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)



        self.to(device)



    def custom_bert_encoder(self, module):
        def custom_forward(*inputs):
            output = module(inputs[0],
                            inputs[1],
                            inputs[2],
                            inputs[3],
                            inputs[4],
                            )
            return output

        return custom_forward

    def forward(self, src, segs, clss, mask_src, mask_cls):
        encoded = self.sentence_encoder(src, segs, clss, mask_src, mask_cls)
        sent_score = self.sentence_predictor(encoded, mask_cls)

        # encoded = checkpoint.checkpoint(
        #     self.custom_bert_encoder(self.sentence_encoder),
        #     src,
        #     segs,
        #     clss,
        #     mask_src,
        #     mask_cls
        # )
        #
        # sent_score = checkpoint.checkpoint(
        #     self.custom_sent_decider(self.sentence_predictor),
        #     encoded,
        #     mask_cls,
        # )

        if self.is_joint:
            sect_scores = self.section_predictor(encoded, mask_cls)
            return sent_score, sect_scores, mask_cls

        else:
            return sent_score, mask_cls


class SentenceEncoder(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(SentenceEncoder, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
        self.ext_transformer_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size,
                                                           args.ext_heads,
                                                           args.ext_dropout, args.ext_layers)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads,
                                     intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_transformer_layer = Classifier(self.bert.model.config.hidden_size)

        if (args.max_pos > 512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(args.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        # if checkpoint is not None:
        #     import pdb;pdb.set_trace()
        #     self.load_state_dict(checkpoint['model'], strict=True)

        # else:
        #     if args.param_init != 0.0:
        #         for p in self.ext_transformer_layer.parameters():
        #             p.data.uniform_(-args.param_init, args.param_init)
        #     if args.param_init_glorot:
        #         for p in self.ext_transformer_layer.parameters():
        #             if p.dim() > 1:
        #                 xavier_uniform_(p)

        # self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)

        # top_vec = checkpoint.checkpoint(
        #     self.custom_sent_decider(self.bert),
        #     src, segs, mask_src
        # )

        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        encoded_sent = self.ext_transformer_layer(sents_vec, mask_cls)
        return encoded_sent


class SentenceExtLayer(nn.Module):
    def __init__(self):
        super(SentenceExtLayer, self).__init__()
        self.wo = nn.Linear(768, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores


class SectionExtLayer(nn.Module):
    def __init__(self):
        super(SectionExtLayer, self).__init__()
        self.wo_2 = nn.Linear(768, 5, bias=True)
        # self.dropout = nn.Dropout(0.5)
        # self.leakyReLu = nn.LeakyReLU()

    def forward(self, x, mask):
        # sent_sect_scores = self.dropout(self.wo_2(x))
        sent_sect_scores = self.wo_2(x)
        sent_sect_scores = sent_sect_scores.squeeze(-1) * mask.unsqueeze(2).expand_as(sent_sect_scores).float()

        return sent_sect_scores


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if (args.max_pos > 512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(args.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if bert_from_extractive is not None:
            # self.bert.model.load_state_dict(dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('sentence_encoder.bert.model')]), strict=True)
            self.bert.model.load_state_dict(dict([(n[28:], p) for n, p in bert_from_extractive.items() if n.startswith('sentence_encoder.bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)


        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight

        if checkpoint is not None:
            # import pdb;pdb.set_trace()
            self.load_state_dict(checkpoint['model'], strict=True)
            # self.decoder.load_state_dict()
            # self.decoder.load_state_dict(dict([(n[8:], p) for n, p in checkpoint['model'].items() if n.startswith('decoder.')]), strict=True)

        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if (args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def custom(self, module):
        def custom_forward(*inputs):
            """

            :param inputs: 0: tgt, 1: top_vec, 2:src
            :type inputs:
            :return:
            :rtype:
            """
            output = module(inputs[0][:, :-1], inputs[1], inputs[2])
            return output[0], output[1]

        return custom_forward

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        # top_vec = self.bert(src, segs, mask_src)
        #
        # # decoder_outputs = checkpoint.checkpoint(
        # #     self.custom(self.decoder), tgt, top_vec, src
        # # )
        # dec_state = self.decoder.init_decoder_state(src, top_vec)
        # # decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        # decoder_outputs = self.decoder(tgt[:, :-1], top_vec, dec_state)
        #
        #
        # return decoder_outputs, None
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
