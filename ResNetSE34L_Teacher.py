#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class ResNetSE_Teacher(nn.Module):
    def __init__(self, block=SEBasicBlock, layers=[3, 4, 6, 3], num_filters=[16, 32, 64, 128], nOut=256, encoder_type='SAP', n_mels=80, log_input=True, **kwargs):
        super(ResNetSE_Teacher, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes   = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels     = n_mels
        self.log_input  = log_input

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 1))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 1))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)

        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion
        elif self.encoder_type == "ASP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion * 2
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        ## If you want analsys, Do self.forward = self.entire_wave2emb or self.entire_wave2emb_div
        # self.forward = self.entire_wave2emb ## for use FlopCountAnalysis (Comment out def forward) 

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def _before_pooling(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.mean(x, dim=2, keepdim=False)
        return x

    def _before_penultimate(self, x):
        if self.encoder_type == "SAP":
            x = x.permute(0,2,1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)
        elif self.encoder_type == "ASP":
            x = x.permute(0,2,1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt( ( torch.sum((x**2) * w, dim=1) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,rh),1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def wave2feat(self, x, max_frame=False):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                if max_frame:
                    audiosize = x.shape[1]
                    max_audio = max_frame*160 +240 # 16kHz defalt
                    if audiosize <= max_audio:
                        import math
                        shortage    = math.floor( ( max_audio - audiosize + 1 ) / 2 )
                        x = F.pad(x, (shortage,shortage), "constant", 0)                
                x = self.torchfb(x)+1e-6
                if self.log_input: x = x.log()
                x = self.instancenorm(x).unsqueeze(1)
                x = x[:,:,:,1:-1]
                x = x.detach()
        return x

    
    def wave2emb(self, wave, max_frame=False):
        feat = self.wave2feat(wave, max_frame)
        late_feat = self._before_pooling(feat)
        emb = self._before_penultimate(late_feat)
        return emb

    def feat2emb(self, feat):
        late_feat = self._before_pooling(feat)
        emb = self._before_penultimate(late_feat)
        return emb

    def wave2feats(self, x, sub_frame, remain_hop=0, max_frame=False):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                if max_frame:
                    audiosize = x.shape[-1]
                    max_audio = max_frame*160 +240 # 16kHz defalt
                    if audiosize <= max_audio:
                        import math
                        shortage    = math.floor( ( max_audio - audiosize + 1 ) / 2 )
                        x = F.pad(x, (shortage,shortage), "constant", 0)

                sub_wave_len = sub_frame*160
                num_sub_waves = (max_audio - 240)//sub_wave_len
                feats = []
                for i, j in enumerate(range(remain_hop, num_sub_waves)):
                    if i == 0:
                        sub_wave = x[:,0:sub_wave_len+120].clone()
                        sub_wave = F.pad(sub_wave, (0,120), "constant", 0)
                    elif i == num_sub_waves-1:
                        sub_wave = x[:,j*sub_wave_len+120:].clone()
                        sub_wave = F.pad(sub_wave, (120,0), "constant", 0)
                    else:
                        sub_wave = x[:,j*sub_wave_len+120:(j+1)*sub_wave_len+120].clone()
                        sub_wave = F.pad(sub_wave, (120,120), "constant", 0)

                    if sub_wave.shape[1] != sub_wave_len+240:
                        print(j, x.size(), sub_wave.size())
                        raise Exception("Wrong sub wave length")

                    sub_wave = self.torchfb(sub_wave)+1e-6
                    if self.log_input: sub_wave = sub_wave.log()
                    sub_wave = self.instancenorm(sub_wave).unsqueeze(1)
                    sub_wave = sub_wave[:,:,:,1:-1]
                    sub_wave = sub_wave.detach()
                    feats.append(sub_wave)
        return feats

    def feats2emb(self, feats): # may not use
        late_feats = []
        for feat in feats:
            late_feat = self._before_pooling(feat)
            late_feats.append(late_feat)
        late_feat = torch.cat(late_feats, dim=-1)
        emb = self._before_penultimate(late_feat)
        return emb

    def div_wave2emb(self, wave, sub_frame, max_frame, num_hop=False, hop_feats=False): # For comparing counter part, num_hop should be larger than one
        if num_hop:
            if num_hop < 1:
                raise Exception(" 'num_hop' should be larger than one")
            elif num_hop > max_frame//sub_frame:
                raise Exception(" Too large 'num_hop'")
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                late_feats = []
                if hop_feats:
                    num_remain_hop = len(hop_feats)
                    if num_hop != num_remain_hop:
                        raise Exception("Wrong number of hop feats")
                    late_feat = hop_feats
                else:
                    num_remain_hop = 0

                feats = self.wave2feats(wave, sub_frame=sub_frame, remain_hop=num_remain_hop, max_frame=max_frame)
                for feat in feats:
                    late_feat = self._before_pooling(feat)
                    late_feats.append(late_feat)
                emb = self._before_penultimate(torch.cat(late_feats, dim=-1)) 

                if num_hop:
                    hop_feats = late_feats[(num_hop*(-1)-1):].clone()
                    return emb, hop_feats
                else:
                    return emb

    def wave2late(self, wave, max_frame=False):
        feat = self.wave2feat(wave, max_frame)
        late_feat = self._before_pooling(feat)
        return late_feat

    def forward(self, x, max_frame=False):
        x = self.wave2emb(x, max_frame)
        return x

    
    def entire_wave2emb_div(self, wave, sub_frame=50, max_frame=200, num_hop=False, hop_feats=False, print_options=False): 
    # def entire_wave2emb_div(self, wave, max_frame, sub_frame=50, num_hop=False, hop_feats=False, print_options=False): 
        import math
        sr = 160
        entire_frame = math.ceil(wave.shape[-1]/sr)
        if entire_frame < max_frame:
            print("--- Processing ---")
        else:
            # import math
            # sr = 160
            if num_hop:
                if num_hop < 1:
                    raise Exception(" 'num_hop' should be larger than one (int) or set to False")
                elif num_hop > max_frame//sub_frame:
                    raise Exception(" Too large 'num_hop'")
            # entire_frame = math.ceil(wave.shape[-1]/sr)
            max_len = max_frame*sr

            if entire_frame < max_frame:
                raise Exception(" 'max frame' should be smaller than wave. However the shape of input wave may be somthing wrong")
            shift_frame = max_frame-(num_hop*sub_frame)
            shift_len = shift_frame*sr
            num_for_loop = math.ceil((entire_frame-max_frame)/shift_frame)+1
            
            if print_options:
                print('entire_frame :', entire_frame, 'shift_frame : ',shift_frame, 'number of the for loops : ',num_for_loop)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    late_feats = []
                    if hop_feats:
                        num_remain_hop = len(hop_feats)
                        if num_hop != num_remain_hop:
                            raise Exception("Wrong number of hop feats")
                        late_feats = hop_feats
                    else:
                        num_remain_hop = 0

                    embs = []

                    frame_ = max_frame

                    end_len = max_frame*sr
                    start_len = num_remain_hop*sub_frame*sr
                    
                    for i, _ in enumerate(range(num_for_loop)):

                        if i == num_for_loop-1: 
                            # last loop
                            end_len = wave.shape[-1]
                            start_len = end_len -max_frame*sr +num_remain_hop*sub_frame*sr
                            sub_wave = wave[:,start_len:]
                        else:
                            sub_wave = wave[:,start_len:end_len]
                        
                        if print_options:
                            print('for loop :',i, 'start index :',start_len, 'end index :',end_len, 'sub wave length :',sub_wave.size())
                        
                        feats = self.wave2feats(sub_wave, sub_frame=sub_frame, max_frame=frame_)
                        for feat in feats:
                            late_feat = self._before_pooling(feat)
                            late_feats.append(late_feat)
                        emb = self._before_penultimate(torch.cat(late_feats, dim=-1)) 

                        if num_hop:
                            embs.append(emb)
                            hop_feats = late_feats[num_hop*(-1):].copy()
                            num_remain_hop = len(hop_feats)
                            if num_hop != num_remain_hop:
                                raise Exception("Wrong number of hop feats")
                            late_feats = hop_feats
                            frame_ = max_frame - num_hop*sub_frame
                        else:
                            late_feats = []
                            embs.append(emb)

                        end_len += shift_len
                        start_len = end_len -max_frame*sr +num_remain_hop*sub_frame*sr                        
            return embs


    # def entire_wave2emb(self, wave, max_frame=300, hop_frame=250, print_options=False): 
    def entire_wave2emb(self, wave, max_frame, hop_frame, print_options=False): 
        import math
        sr = 160
        entire_frame = math.ceil(wave.shape[-1]/sr)
        max_len = max_frame*sr
        
        if entire_frame < max_frame:
            print("--- Processing ---")
            
        else:
            shift_frame = max_frame-hop_frame
            shift_len = shift_frame*sr
            num_for_loop = math.ceil((entire_frame-max_frame)/shift_frame)+1
            
            if print_options:
                print('entire_frame :', entire_frame, 'shift_frame : ',shift_frame, 'number of the for loops : ',num_for_loop)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    embs = []

                    end_len = max_frame*sr
                    start_len = 0
                    
                    for i, _ in enumerate(range(num_for_loop)):

                        if i == num_for_loop-1: 
                            # last loop
                            start_len = wave.shape[-1] -max_frame*sr
                            sub_wave = wave[:,start_len:]
                        else:
                            sub_wave = wave[:,start_len:end_len]
                        
                        if print_options: 
                            print('for loop :',i, 'start index :',start_len, 'end index :',end_len, 'sub wave length :',sub_wave.size())
                        
                        end_len += shift_len
                        start_len = end_len -max_frame*sr
                        
                        feat = self.wave2feat(sub_wave)
                        feat = self._before_pooling(feat)
                        emb = self._before_penultimate(feat) 
                        embs.append(emb)
            return embs


if __name__=="__main__":
    # batch_size, num_frames, feat_dim = 1, 3000, 80
    batch_size, second = 1, 30
    x = torch.randn(batch_size, int(second*16000))
    # x_tp = x.transpose(1, 2)

    model = ResNetSE_Teacher(nOut = 256)

    # model.eval()
    # embs = model(x, print_options=True)
    # exit()

    torch.set_num_threads(1)
    import timeit
    model.eval()
    number = 10
    end_start = timeit.timeit(stmt='model(x)', globals=globals(), number=number)
    print('CPU Time :',end_start/number*1000, 'ms') 
    # exit()

    model.eval()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    embs = model(x)
    end.record()
    torch.cuda.synchronize()
    print('GPU Time :',start.elapsed_time(end), 'ms')
    # exit()

    from fvcore.nn import FlopCountAnalysis
    # self.forward = self.entire_wave2emb ## for use FlopCountAnalysis (Comment out def forward)
    model.eval()
    flops = FlopCountAnalysis(model, x)
    print(flops.total())