import numpy as np
import torch, math
import torch.nn as nn
from utils.ContrastLoss import SupConLoss

class LTHNetLoss(nn.Module):
    """
    LTHNet loss function.

    Args
        epoch (float): the current epoch for calculating alpha (balanced or not).
        beta (float): class-balanced hyper-parameter
        num_per_class mapping: number of samples for each class.
        gamma: cross-entropy-loss vs. class-balanced-loss
    """

    def __init__(self):
        super(LTHNetLoss, self).__init__()
        print('Long-Tailed Hashing Loss works!')

    def forward(self, hashcodes, assignments, targets, device, beta, mapping ):
        # eg. mapping['0']=500, mapping['1']=100, etc.
        # -------------------------------------------------------------
        batch_size = assignments.size(0)
        num_classes = assignments.size(1)
        code_length = hashcodes.size(1)

        # -------------------------------------------------------------
        # mini-batch cross-entropy loss between assignments and targets: softmax-log-NLL-average
        # pointwise loss
        loss_cross_entropy = torch.sum(- torch.log(assignments) * targets) / batch_size

        # balanced factor (class)
        balance_factor = torch.zeros([num_classes])
        for j in range(len(mapping)):
            balance_factor[j] = (1 - beta) / (1 - beta ** mapping[str(j)])
        balance_factor = balance_factor / torch.max(balance_factor)

        # class-balanced loss
        weights = torch.Tensor.repeat(balance_factor, [batch_size, 1]).to(device)
        print(assignments)
        loss_class_balanced = torch.sum(- torch.log(assignments) * targets * weights) / batch_size

        # gradual learning
        # alpha = 1 - (epoch * 1.0 / maxIter) ** 2

        # overall loss
        # loss = alpha * loss_cross_entropy + (1 - alpha) * (gamma * loss_class_balanced)

        return loss_class_balanced


class Combiner:
    def __init__(self, device, max_epoch):
  
        self.device = device
        self.epoch_number =  max_epoch
        self.softmaxLoss =  LTHNetLoss()
        self.contrastLoss =  SupConLoss()
        self.initilize_all_parameters()

    def initilize_all_parameters(self):
        self.alpha = 0.2
        if self.epoch_number in [90, 180]:
            self.div_epoch = 100 * (self.epoch_number // 100 + 1)
        else:
            self.div_epoch = self.epoch_number
    
    def reset_epoch(self, epoch):
        self.epoch = epoch



    def bbn_mix_loss(self, model, image, label, meta, prototypes,beta,mapping):

        image_a, image_b = image ,meta["sample_image"].to(self.device)
        label_a, label_b = label, meta["sample_label"].to(self.device)
        x1 = image_a[0].to(self.device)
        x2 = image_a[1].to(self.device)
        image_a = torch.cat([x1,x2],dim=0)
        label_a = torch.cat([label_a,label_a],dim = 0)
        imbalanced_feature = model(image_a, False, prototypes) 
        hash_codes, assignments, direct_feature = model(image_b, True, prototypes)
        loss_contrast = self.contrastLoss(imbalanced_feature,label_a,imbalanced_feature,label_a)
        loss_classification = self.softmaxLoss(hash_codes,assignments,label_b, self.device, beta,mapping)

        l = 1 - ((self.epoch - 1) / self.div_epoch) ** 2  # parabolic decay
        #l = 0.5  # fix
        #l = math.cos((self.epoch-1) / self.div_epoch * math.pi /2)   # cosine decay
        #l = 1 - (1 - ((self.epoch - 1) / self.div_epoch) ** 2) * 1  # parabolic increment
        #l = 1 - (self.epoch-1) / self.div_epoch  # linear decay
        #l = np.random.beta(self.alpha, self.alpha) # beta distribution
        #l = 1 if self.epoch <= 120 else 0  # seperated stage

        #loss = l * loss_contrast+ (1 - l) * loss_classification
        loss = loss_contrast
       # print("l is", l)
        
        
        return loss

