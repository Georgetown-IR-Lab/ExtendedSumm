import copy

import torch
import torch.nn as nn
# from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
from torch.utils import checkpoint
# from torch.autograd import Variable
from transformers import BertModel, BertConfig, LongformerModel

# from models.pointer_generator.PG_transformer import PointerGeneratorTransformer
from models.decoder import TransformerDecoder
from models.encoder import ExtTransformerEncoder
from models.optimizers import Optimizer
from models.uncertainty_loss import UncertaintyLoss


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
    gen_func = nn.LogSoftmax(dim=1)
    generator = nn.Sequential(nn.Linear(dec_hidden_size, vocab_size), gen_func)
    generator.to(device)
    return generator


class Bert(nn.Module):
    def __init__(self, large, model_name, temp_dir, finetune=False):
        super(Bert, self).__init__()

        if model_name == 'bert':
            if (large):
                self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
            else:
                self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        elif model_name == 'scibert':
            self.model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=temp_dir)

        elif model_name == 'longformer':
            if large:
                self.model = LongformerModel.from_pretrained('allenai/longformer-large-4096', cache_dir=temp_dir)
            else:
                self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096', cache_dir=temp_dir)

        self.model_name = model_name
        self.finetune = finetune


    def forward(self, x, segs, mask_src, mask_cls, clss):
        if (self.finetune):

            if self.model_name =='bert' or self.model_name =='scibert':
                top_vec, _ = self.model(x, attention_mask=mask_src, token_type_ids=segs)

            elif self.model_name=='longformer':
                global_mask = torch.zeros(mask_src.shape, dtype=torch.long, device='cuda').unsqueeze(0)
                global_mask[:,:,clss.long()] = 1
                global_mask = global_mask.squeeze(0)
                top_vec, _ = self.model(x, attention_mask=mask_src.long(), global_attention_mask=global_mask)

        else:
            self.eval()
            with torch.no_grad():
                if self.model_name == 'bert' or self.model_name == 'scibert':
                    top_vec, _ = self.model(x, attention_mask=mask_src, token_type_ids=segs)

                elif self.model_name == 'longformer':
                    global_mask = torch.zeros(mask_src.shape, dtype=torch.long, device='cuda').unsqueeze(0)
                    global_mask[:, :, clss.long()] = 1
                    global_mask = global_mask.squeeze(0)

                    top_vec, _ = self.model(x, attention_mask=mask_src.long(), global_attention_mask=global_mask)

        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint, is_joint=True, rg_predictor=False):
        super(ExtSummarizer, self).__init__()
        self.is_joint = is_joint
        self.sentence_encoder = SentenceEncoder(args, device, checkpoint)
        self.sentence_predictor = SentenceExtLayer()
        self.rg_predictor = rg_predictor
        if is_joint:
            self.uncertainty_loss = UncertaintyLoss()
            self.section_predictor = SectionExtLayer()
            self.loss_sentence_picker = torch.nn.BCELoss(reduction='none')
            self.loss_section_predictor = torch.nn.CrossEntropyLoss(reduction='none')


        if not is_joint and not rg_predictor:
            self.loss_sentence_picker = torch.nn.BCELoss(reduction='none')

        if rg_predictor:
            self.loss_sentence_picker = torch.nn.MSELoss(reduction='none')

        self.args = args

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



    def forward(self, src, segs, clss, mask_src, mask_cls, sent_bin_labels, sent_sect_labels, is_inference=False, return_encodings=False):

        encoded = self.sentence_encoder(src, segs, clss, mask_src, mask_cls)
        sent_score = self.sentence_predictor(encoded, mask_cls)

        if self.is_joint:
            sect_scores = self.section_predictor(encoded, mask_cls)

            if self.args.alpha_mtl != -1:
                loss_sent = self.loss_sentence_picker(sent_score, sent_bin_labels.float())
                loss_sent = ((loss_sent * mask_cls.float()).sum() / mask_cls.sum(dim=1)).sum()
                if not is_inference:
                    loss_sent = loss_sent / loss_sent.numel()

                loss_sect = self.loss_section_predictor(sect_scores.permute(0, 2, 1), sent_sect_labels)
                loss_sect = ((loss_sect * mask_cls.float()).sum() / mask_cls.sum(dim=1)).sum()
                if not is_inference:
                    loss_sect = (loss_sect / loss_sect.numel())

                loss = (self.args.alpha_mtl) * loss_sent + (1-(self.args.alpha_mtl)) * loss_sect

            else:
                loss, loss_sent, loss_sect = self.uncertainty_loss(sent_score, sent_bin_labels, sect_scores, sent_sect_labels, mask=mask_cls)

            return sent_score, sect_scores, mask_cls, loss, loss_sent, loss_sect

        else:
            if not self.rg_predictor:
                try:
                    loss = self.loss_sentence_picker(sent_score, sent_bin_labels.float())
                except:
                    import pdb;pdb.set_trace()
                loss = ((loss * mask_cls.float()).sum() / mask_cls.sum(dim=1)).sum()
                if not is_inference:
                    loss = loss / loss.numel()

            else:
                loss = self.loss_sentence_picker(sent_score, sent_bin_labels.float())
                loss = ((loss * mask_cls.float()).sum() / mask_cls.sum(dim=1)).sum()
                if not is_inference:
                    loss = loss / loss.numel()

            if return_encodings:
                return sent_score, mask_cls, loss, None, None, encoded

            return sent_score, mask_cls, loss, None, None




class SentenceEncoder(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(SentenceEncoder, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.model_name, args.temp_dir, args.finetune_bert)
        self.ext_transformer_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size,
                                                           args.ext_heads,
                                                           args.ext_dropout, args.ext_layers)

        if args.max_pos > 512 and args.model_name=='bert':
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            import pdb;pdb.set_trace()

            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(args.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings


        if args.max_pos > 4096 and args.model_name=='longformer':
            my_pos_embeddings = nn.Embedding(args.max_pos+2, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:4097] = self.bert.model.embeddings.position_embeddings.weight.data[:-1]
            my_pos_embeddings.weight.data[4097:] = self.bert.model.embeddings.position_embeddings.weight.data[1:args.max_pos +2 - 4096]
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings


        self.sigmoid = nn.Sigmoid()

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src, mask_cls, clss)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
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

    def forward(self, x, mask):
        sent_sect_scores = self.wo_2(x)
        sent_sect_scores = sent_sect_scores.squeeze(-1) * mask.unsqueeze(2).expand_as(sent_sect_scores).float()

        return sent_sect_scores


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.model_name, args.temp_dir, args.finetune_bert)
        self.sentence_encoder = SentenceEncoder(args, device, checkpoint)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
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
            self.load_state_dict(checkpoint['model'], strict=True)
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
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src, mask_cls, clss)

        # src = src[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
        # src = src * mask_cls[:, :, None].float()
        # encoded_sent = self.ext_transformer_layer(sents_vec, mask_cls)

        dec_state = self.decoder.init_decoder_state(src, top_vec)
        tgt = torch.rand(size=(1, 9, 768))
        decoder_outputs, attn_enc_dec, state = self.decoder(tgt[:, :-1], top_vec, clss, dec_state)
        return decoder_outputs, attn_enc_dec, None







