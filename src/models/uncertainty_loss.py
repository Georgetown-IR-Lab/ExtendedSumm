import torch
import torch.nn as nn


class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
        self.bin_loss = nn.BCELoss(reduction='none')
        self.loss_sect = nn.CrossEntropyLoss(reduction='none')

        self.alpha = nn.Parameter(torch.tensor([0, 0], requires_grad=True, dtype=torch.float32).cuda()) # Task weighting that should be learnt...


    def forward(self, input_sent, target_sent, input_sect, target_sect, mask=None):
        loss_sent = self.bin_loss(input_sent, target_sent.float())
        loss_sent = (loss_sent * mask.float()).sum()
        loss_sent = (loss_sent / loss_sent.numel())

        loss_sect = self.loss_sect(input_sect.permute(0, 2, 1), target_sect)
        loss_sect = (loss_sect * mask.float()).sum()
        loss_sect = (loss_sect / loss_sect.numel())

        mtl_loss = ((loss_sent / torch.exp(self.alpha[0])) + self.alpha[0]) \
               + ((loss_sect / torch.exp(self.alpha[1])) + self.alpha[1])


        return mtl_loss, loss_sent, loss_sect