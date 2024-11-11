import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy, tqdm, sys, time, soundfile

from loss import *
# from ECAPATDNN512 import *
# from ECAPATDNN512_Teacher import *
# from ResNetSE34L import *
from ResNet_FT import *
from ResNetSE34L_Teacher import *
from tools import *
import dino_utils as utils

class SpeakerNet(nn.Module):
    def __init__(self, **kwargs):
        super(SpeakerNet, self).__init__();
        # self.backbone = ECAPA_TDNN(**kwargs) # Speaker encoder 
        self.backbone = ResNetSE(**kwargs) # Speaker encoder 
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

class TeacherNet(nn.Module):
    def __init__(self, **kwargs):
        super(TeacherNet, self).__init__();
        # self.backbone = ECAPA_TDNN_Teacher(**kwargs) # Speaker encoder 
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

class model(nn.Module):
    def __init__(self, niter_per_ep=1092009, **kwargs): ##1240652, 1092009:Vox2, 148642:Vox1
        super(model, self).__init__()
        self.TeacherNet = TeacherNet(**kwargs).cuda()
        self.SpeakerNet = SpeakerNet(**kwargs).cuda()

        # self.SpeakerNet.load_state_dict(self.TeacherNet.state_dict())
        # student_state = self.SpeakerNet.state_dict()
        # for name, param in self.TeacherNet.state_dict().items():
        #     origname = name
        #     if name not in student_state:
        #         print("%s is not in the student."%origname)
        #     elif student_state[name].size() != self.TeacherNet.state_dict()[origname].size():
        #         raise Exception("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, student_state[name].size(), self.TeacherNet.state_dict()[origname].size()))
        #     student_state[name].copy_(param)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.TeacherNet.parameters():
            p.requires_grad = False
        print(f"Student and Teacher are built.")

        # synchronize batch norms (if any)
        # if utils.has_batchnorms(self.SpeakerNet):
        #     self.SpeakerNet = nn.SyncBatchNorm.convert_sync_batchnorm(self.SpeakerNet)
        # self.SpeakerNet = nn.parallel.DistributedDataParallel(self.SpeakerNet, device_ids=[kwargs['gpu']])

        # params_groups = list(self.SpeakerNet.parameters())
        # if kwargs['optimizer'] == "adamw":
        #     self.optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        # elif kwargs['optimizer'] == "adam":
        #     self.optimizer = torch.optim.Adam(params_groups, betas = (0.9, 0.999), eps=1e-06, amsgrad=False)    
        # elif kwargs['optimizer'] == "sgd":
        #     self.optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
        # elif kwargs['optimizer'] == "lars":
        #     self.optimizer = utils.LARS(params_groups)

        # print("Model para number = %.2f"%(sum(param.numel() for param in self.SpeakerNet.backbone.parameters()) / 1000 / 1000))
        # print("TeacherNet para number = %.2f"%(sum(param.numel() for param in self.TeacherNet.backbone.parameters()) / 1000 / 1000))

        # # niter_per_ep = 148642 ##length of VoxCeleb1 niter_per_ep = 1092009 ##length of VoxCeleb2
        # self.lr_schedule = utils.cosine_scheduler(
        #     kwargs['lr'],
        #     kwargs['lr_min'],
        #     kwargs['max_epoch'], niter_per_ep,
        #     warmup_epochs=kwargs['warmup_epochs'],
        #     start_warmup_value=1e-8,
        # )
        # self.wd_schedule = utils.cosine_scheduler(
        #     kwargs['weight_decay'],
        #     kwargs['weight_decay_end'],
        #     kwargs['max_epoch'], niter_per_ep,
        # )
        # self.warmup_epochs = kwargs['warmup_epochs']
        # self.clip_grad = kwargs['clip_grad']

        # self.max_frames = kwargs['max_frames']
        # self.sub_frames = kwargs['sub_frames']
        # self.max_epoch = kwargs['max_epoch']

    def train_network(self, loader, epoch):
        stepsize = loader.batch_size;
        epoch = epoch-1

        self.train()

        loss_total, mse, n_loss, top1 = 0, 0, 0, 0 
        time_start = time.time()

        for counter, (data, label) in enumerate(loader, start = 1):
            iters = (len(loader) * epoch + counter)*stepsize  # global training iteration
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.lr_schedule[iters]
                param_group["lr"] = lr
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = self.wd_schedule[iters]

            self.zero_grad()
            data = data.reshape(-1,data.size()[-1]).cuda() 
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    outp_t = self.TeacherNet.backbone.wave2late(data)
            # import random
            # p = random.random()
            # if p < 0.5:
            #     sub_frame = 50
            # else:
            #     sub_frame = 100
            
            sub_frame = 50

            outp_s, mse_loss = self.SpeakerNet.backbone._compute_loss(data, outp_t, sub_frame=sub_frame, max_frame=self.max_frames)
            nloss, prec1 = self.SpeakerNet.head.forward(outp_s, torch.LongTensor(label).cuda()) # input segment and the output speaker embedding
            
            total_loss = nloss + mse_loss

            if not math.isfinite(total_loss.item()):
                print("Loss is {}, stopping training".format(total_loss.item()), force=True)
                sys.exit(1)
        
            param_norms = None
            total_loss.backward()

            if self.clip_grad:
                param_norms = utils.clip_gradients(self.SpeakerNet, self.clip_grad)
            if epoch < self.warmup_epochs:
                utils.exclude_cancel_gradients_with_name(epoch, self.SpeakerNet, self.max_epoch, 'backbone.mhsa')

            self.optimizer.step()
            loss_it = total_loss.detach().cpu().numpy()
            mse_it = mse_loss.detach().cpu().numpy()
            nloss_it = nloss.detach().cpu().numpy()
            loss_total    += loss_it
            mse           += mse_it
            n_loss        += nloss_it
            top1          += prec1

            # logging
            torch.cuda.synchronize()

            time_used = time.time() - time_start # Time for this epoch
            sys.stderr.write("[%2d] Lr: %6f, %.2f%% (est %.1f mins), Mean Loss %.3f (NLoss %.3f), (MSELoss %.3f), Mean ACC: %.2f%% \r" %\
            (epoch, lr, 100 * (counter / loader.__len__()), time_used * loader.__len__() / counter / 60, loss_total/counter, nloss_it, mse_it, top1/counter))
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
            audio = audio[int(320):int(320)+160*300]
            
            feat = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()
            with torch.no_grad():
                # ref_feat = self.SpeakerNet.forward(feat).detach().cpu()
                # ref_feat = self.TeacherNet.forward(feat).detach().cpu()
                # ref_feat = self.SpeakerNet.backbone.forward(feat, max_frame=300).detach().cpu()
                ref_feat = self.SpeakerNet.backbone.div_wave2emb(feat, sub_frame=50, max_frame=200).detach().cpu() ## 9/8
                # ref_feat = self.TeacherNet.backbone.div_wave2emb(feat, sub_frame=100, max_frame=300).detach().cpu()
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

    # def evaluate_network(self, val_list, val_path, **kwargs): ## Add TTA
    #     eval_list = val_list
    #     eval_path = val_path
    #     self.eval()
    #     files = []
    #     embeddings = {}
    #     lines = open(eval_list).read().splitlines()
    #     for line in lines:
    #         files.append(line.split()[1])
    #         files.append(line.split()[2])
    #     setfiles = list(set(files))
    #     setfiles.sort()

    #     for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
    #         audio, _  = soundfile.read(os.path.join(eval_path, file))
    #         # Full utterance
    #         data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()

    #         # Spliited utterance matrix
    #         max_audio = 400 * 160 + 240
    #         if audio.shape[0] <= max_audio:
    #             shortage = max_audio - audio.shape[0]
    #             audio = numpy.pad(audio, (0, shortage), 'wrap')
    #         feats = []
    #         startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
    #         for asf in startframe:
    #             feats.append(audio[int(asf):int(asf)+max_audio])
    #         feats = numpy.stack(feats, axis = 0).astype(numpy.float)
    #         data_2 = torch.FloatTensor(feats).cuda()
    #         # Speaker embeddings
    #         with torch.no_grad():
    #             embedding_1 = self.SpeakerNet.forward(data_1)
    #             embedding_1 = F.normalize(embedding_1, p=2, dim=1)
    #             embedding_2 = self.SpeakerNet.forward(data_2)
    #             embedding_2 = F.normalize(embedding_2, p=2, dim=1)
    #         embeddings[file] = [embedding_1, embedding_2]
    #     scores, labels  = [], []

    #     for line in lines:          
    #         embedding_11, embedding_12 = embeddings[line.split()[1]]
    #         embedding_21, embedding_22 = embeddings[line.split()[2]]
    #         # Compute the scores
    #         score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
    #         score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
    #         score = (score_1 + score_2) / 2
    #         score = score.detach().cpu().numpy()
    #         scores.append(score)
    #         labels.append(int(line.split()[0]))
            
    #     # Coumpute EER and minDCF
    #     EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    #     fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    #     minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

    #     return EER, minDCF

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
                print("%s is not in the studnent model."%origname)
            elif self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, student model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
            self_state[name].copy_(param)
            name = name.replace("SpeakerNet.", "TeacherNet.")
            if name not in self_state:
                print("%s is not in the teacher model."%origname)
            elif self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, teacher model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
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
        print("Pre-trained Model loaded to student!")

        for name, param in loaded_state.items():
            if load_model_name in name:
                origname = name
                name = name.replace(load_model_name, "TeacherNet.backbone.")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue                    
                if self_state[name].size() != loaded_state[origname].size():
                    print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                    continue
                if 'backbone.' in name:
                    self_state[name].copy_(param)
        print("Pre-trained Model loaded to teacher!")       
        print("Model init complete!")

