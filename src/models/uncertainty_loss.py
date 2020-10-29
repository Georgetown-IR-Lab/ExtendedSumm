import torch
import torch.nn as nn


class UncertaintyLoss(nn.Module):
    def __init__(self, rg_predictor=True):
        super(UncertaintyLoss, self).__init__()
        self.rg_predictor = rg_predictor
        if not self.rg_predictor:
            self.sentence_picker_loss = nn.BCELoss(reduction='none')
        else:
            self.sentence_picker_loss = nn.MSELoss(reduction='none')

        self.loss_sect = nn.CrossEntropyLoss(reduction='none')
        self._sigmas_sq = nn.Parameter(torch.tensor([0.50, 0.50], requires_grad=True, dtype=torch.float32).cuda()) # Task weighting that should be learnt...

        # torch.nn.init.uniform_(self._sigmas_sq, a=0.2, b=1.0)
        # self._sigmas_sq[0] = 0.5
        # self._sigmas_sq[1] = 0.5

    def mtl_loss(self, _loss_list):
        factor = torch.div(1.0, torch.mul(2.00, self._sigmas_sq[0]))
        loss = torch.add(torch.mul(factor, _loss_list[0]), torch.log(self._sigmas_sq[0]))
        for i in range(1, len(_loss_list)):
            factor = torch.div(1.0, self._sigmas_sq[i])
            loss = torch.add(loss, torch.add(torch.mul(factor, _loss_list[i]), torch.log(self._sigmas_sq[i])))
        return loss

    def mtl_loss_simple(self, _loss_list):

        return (self._sigmas_sq[0] * _loss_list[0]) + (self._sigmas_sq[1] * _loss_list[1])


    def forward(self, input_sent, target_sent, input_sect, target_sect, mask=None):

        if not self.rg_predictor:
            loss_sent = self.sentence_picker_loss(input_sent, target_sent.float())
            loss_sent = (loss_sent * mask.float()).sum()
            loss_sent = (loss_sent / loss_sent.numel())
        else:
            loss_sent = self.sentence_picker_loss(input_sent, target_sent.float())
            loss_sent = ((loss_sent * mask.float()).sum() / mask.sum(dim=1)).sum()
            loss_sent = (loss_sent / loss_sent.numel())

        loss_sect = self.loss_sect(input_sect.permute(0, 2, 1), target_sect)
        loss_sect = (loss_sect * mask.float()).sum()
        loss_sect = (loss_sect / loss_sect.numel())


        mtl_loss = self.mtl_loss_simple([loss_sent, loss_sect])


        return mtl_loss, loss_sent, loss_sect