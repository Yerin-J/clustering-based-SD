import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy, tqdm, sys, time, soundfile

from loss import *
from ResNetSE34L_Teacher import *
from tools import *
import dino_utils as utils

class SpeakerNet(nn.Module):
    def __init__(self, **kwargs):
        super(SpeakerNet, self).__init__();
        self.backbone = ResNetSE_Teacher(**kwargs) # Speaker encoder 
        self.head = LossFunction(
                nOut = kwargs['nOut'],
                n_class = kwargs['n_class'],
                margin = kwargs['margin'],
                scale = kwargs['scale']
                ) # Classification layer
    def forward(self, data, label=None):

        data    = data.reshape(-1,data.size()[-1]).cuda() 
        outp    = self.backbone.forward(data)

        if label == None:
            return outp
        else:
            nloss, prec1 = self.head.forward(outp,label)
            return nloss, prec1

class model_pr(nn.Module):
    def __init__(self, niter_per_ep=1092009, **kwargs): ##1240652, 1092009:Vox2, 148642:Vox1
        super(model_pr, self).__init__()
        self.SpeakerNet = SpeakerNet(**kwargs)

    def train_network(self, loader, epoch):
        stepsize = loader.batch_size;
        epoch = epoch-1

        self.train()

        loss_total, top1 = 0, 0
        time_start = time.time()

        for counter, (data, label) in enumerate(loader, start = 1):
            iters = (len(loader) * epoch + counter)*stepsize  # global training iteration
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.lr_schedule[iters]
                param_group["lr"] = lr
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = self.wd_schedule[iters]

            self.zero_grad()
            nloss, prec1 = self.SpeakerNet.forward(data.cuda(), torch.LongTensor(label).cuda()) # input segment and the output speaker embedding
            
            if not math.isfinite(nloss.item()):
                print("Loss is {}, stopping training".format(nloss.item()), force=True)
                sys.exit(1)
        
            param_norms = None
            nloss.backward()

            if self.clip_grad:
                param_norms = utils.clip_gradients(self.SpeakerNet, self.clip_grad)

            self.optimizer.step()
            loss_it = nloss.detach().cpu().numpy()
            loss_total    += loss_it
            top1          += prec1

            # logging
            torch.cuda.synchronize()

            time_used = time.time() - time_start # Time for this epoch
            sys.stderr.write("[%2d] Lr: %6f, %.2f%% (est %.1f mins), Mean Loss %.3f (Loss %.3f), Mean ACC: %.2f%% \r" %\
            (epoch, lr, 100 * (counter / loader.__len__()), time_used * loader.__len__() / counter / 60, loss_total/counter, loss_it, top1/counter))
            sys.stderr.flush()

        sys.stdout.write("\n")

        return loss_total/counter, top1/counter, lr

    def evaluate_network(self, val_list, val_path, **kwargs):
        self.eval()
        files, feats = [], {}
        for line in open(val_list).read().splitlines():
            data = line.split()
            files.append(data[1])
            files.append(data[2])
        setfiles = list(set(files))
        setfiles.sort()  # Read the list of wav files
        for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
            audio, _ = soundfile.read(os.path.join(val_path, file))
            feat = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()
            with torch.no_grad():
                ref_feat = self.SpeakerNet.forward(feat).detach().cpu()
            feats[file]     = ref_feat # Extract features for each data, get the feature dict
        scores, labels  = [], []
        for line in open(val_list).read().splitlines():
            data = line.split()
            ref_feat = F.normalize(feats[data[1]].cuda(), p=2, dim=1) # feature 1
            com_feat = F.normalize(feats[data[2]].cuda(), p=2, dim=1) # feature 2
            score = numpy.mean(torch.matmul(ref_feat, com_feat.T).detach().cpu().numpy()) # Get the score
            scores.append(score)
            labels.append(int(data[0]))
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)
        return EER, minDCF

    
    def save_network(self, path): # Save the model
        save_dict = {
            'network': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        utils.save_on_master(save_dict, path)

    def load_network(self, path): # Load the parameters of the pretrain model
        self_state = self.state_dict()
        checkpoint = torch.load(path, map_location="cpu")
        print("Ckpt file %s loaded!"%(path))

        if 'network' not in checkpoint.keys():
            loaded_state = checkpoint
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer loaded!")
            loaded_state = checkpoint['network']

        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                raise Exception("%s is not in the model."%origname)
            elif self_state[name].size() != loaded_state[origname].size():
                raise Exception("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
            self_state[name].copy_(param)
        print("Model loaded!")


    def init_network(self, path): # Load the parameters of the pretrain model with different Stage
        self_state = self.state_dict()
        checkpoint = torch.load(path, map_location="cpu")
        print("Ckpt file %s loaded!"%(path))

        if 'network' not in checkpoint.keys():
            loaded_state = checkpoint
        else:
            loaded_state = checkpoint['network']

        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                print("%s is not in the model."%origname)
            elif self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
            self_state[name].copy_(param)
        print("Model init complete!")


    def init_download_network(self, path): # Load the parameters of the pretrain model with different name
        self_state = self.state_dict()
        checkpoint = torch.load(path, map_location="cpu")
        print("Ckpt file %s loaded!"%(path))

        if 'network' not in checkpoint.keys():
            loaded_state = checkpoint
        else:
            loaded_state = checkpoint['network']

        # load_model_name = "speaker_encoder." # (Pre-trined ECAPA-TDNN)
        load_model_name = "__S__." # (Pre-trined ResNet)
        # load_model_name = "" # custom        
        for name, param in loaded_state.items():
            if load_model_name in name:
                origname = name
                name = name.replace(load_model_name, "SpeakerNet.backbone.")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue                    
                if self_state[name].size() != loaded_state[origname].size():
                    print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                    continue
                if 'backbone.' in name:
                    self_state[name].copy_(param)
        print("Pre-trained Model loaded!")