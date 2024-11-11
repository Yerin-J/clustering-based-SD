#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pprint import pprint

import fastcluster
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

from VBx.diarization_lib import twoGMMcalib_lin, AHC
from VB_diarization import VB_diarization

import diarization_sb as diar

from utils.meta_utils import make_wavscp2
from utils.io_utils import load_dict_txt, linecount
from kaldi_io.arkio import read_kaldi_arkscp, write_kaldi_binary
from kaldi_io.data_utils import load_dict_segments

from sklearn.manifold import TSNE

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

from hparam_is24 import Hyperparams as hp

from model import model_pr ################################################################################## Proposed model
# from model_proposed import model ########################################################################## Proposed model with fine_tuning

from PldaVariant2 import compute_css_pairwise, length_norm

##refinement
from scipy.ndimage import gaussian_filter

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))


batch_mode = False
mbs = 64

## =============  Hyperparameters  ============= ##
# db = 'dev' # 'dev', 'test'
db = 'test' # 'dev', 'test'

score_type = 'cos' # 'cos', 'plda'

cluster_type = 'spec' #'spec' # 'ahc', 'spec', 'ahc+VBx'
# cluster_type = 'ahc+VBx' # 'ahc', 'spec'

num_oracle_spks = None
fast_clustering = True

p_val = 0.05 # for spectral clustring

use_external_sad = True

## =============  Dataset-specific  ============= ##

## =============  Scoring options  ============= ##
ignore_overlaps = False
forgiveness_collar = 0.25
plot_der_thres, plot_jer_thres = 5.0, 10.0



if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu_id', type=str, default='')
  parser.add_argument("--overwrite", dest="overwrite", action='store_true')
  parser.add_argument('--segLen', required=False, type=int, default=200, help='segment length (in frames)')
  parser.add_argument('--segSht', required=False, type=int, default=50, help='segment shift (in frames)')
  parser.set_defaults(overwrite=False)
  args = parser.parse_args()

  segLen = args.segLen
  segSht = args.segSht
  MIN_SEG_DUR = 0.01 # in seconds


  ## Files and directories to load/save inputs/outputs
  reclist = f'reclist/voxconverse/{db}.txt'
  db_base = f'./voxconverse'
  wav_dir = f'{db_base}/voxconverse_{db}_wav'
  ref_rttm_dir = f'{db_base}/rttms/{db}' # can be hardcoded

  exp_dir = f'exp_proposed/voxconverse' ###################################################################################################### 저장 경로 설정
  # exp_dir = f'exp_proposed_w_fine_tuning/voxconverse' ###################################################################################### 저장 경로 설정

  if use_external_sad:
    vad_tag = 'BUT_SAD'
    exp_dir = f'{exp_dir}/{vad_tag}/{db}'
    vad_dir = f'VAD/voxconverse/final_system/labs_{db}' # can be hardcoded
    assert os.path.isdir(vad_dir), vad_dir
  else:
    vad_tag = ValueError
    exp_dir = f'{exp_dir}/{vad_tag}/{db}'
    vad_dir = f'{exp_dir}/vads'
  os.makedirs(exp_dir, exist_ok=True)
  seg_dir = f'{exp_dir}/segments/{segLen}_{segSht}'
  emb_dir = f'{exp_dir}/embeddings/{segLen}_{segSht}'
  rttm_dir = f'{exp_dir}/rttms/{score_type}_{cluster_type}/{segLen}_{segSht}'
  result_file = f'{exp_dir}/results/{score_type}_{cluster_type}_{segLen}_{segSht}'
  os.makedirs(seg_dir, exist_ok=True)
  os.makedirs(emb_dir, exist_ok=True)
  os.makedirs(rttm_dir, exist_ok=True)
  os.makedirs(os.path.dirname(result_file), exist_ok=True)

  ref_rttms_cat = ref_rttm_dir+'.rttm'
  if not os.path.isfile(ref_rttms_cat):
    os.system(f'cat {ref_rttm_dir}/*.rttm > {ref_rttms_cat}')
  # exit()


  ## Prepare wav.scp
  wav_scp = f'{exp_dir}/wav_{db}.scp'
  dir_level = 1 if db=="dev" else 1
  if not os.path.isfile(wav_scp):
    make_wavscp2(wav_dir, extension='wav', dir_level=dir_level, 
      delim='dummy', wavscp=wav_scp, sort=True)
  ## Prepare reclist
  if not os.path.isfile(reclist):
    with open(reclist, 'wt') as f:
      for line in open(wav_scp):
        uttid = line.rstrip().split(' ')[0]
        f.write(f'{uttid}\n')
  ## wav.scp must be filtered out by reclist
  recids = [l.strip().split(' ')[0] for l in open(reclist)]
  os.rename(wav_scp, wav_scp+'.tmp')
  with open(wav_scp, 'wt') as f:
    for line in open(wav_scp+'.tmp'):
      uttid, wpath = line.rstrip().split(' ')
      for recid in recids:
        if recid in uttid:
          f.write(f'{recid} {wpath}\n')
          break
  os.remove(wav_scp+'.tmp')
  # exit()


  ## Check if embedding extraction was finished
  missing_embeddings = any(
    not os.path.isfile('%s/%s.ark' % (emb_dir, l.rstrip().split(' ')[0]))
    for l in open(reclist)
  )

  ## Load speaker embedding model
  if missing_embeddings:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    ## ---------------------------------------------------------------------- ##

    ## STFT configs
    fs = hp.mel_opts_torch['fs']
    winLen = hp.stft_opts_torch['win_length']
    winSht = hp.stft_opts_torch['hop_length']

    ## Load pretrained speaker embedding model
    n_outs = {'spk':5994}
    
    # ResNet
    ####################################################################################################################################### 실습 #2
    # Conventional model
    model_dict = torch.load('<< fastres_model000000069.model 경로 >>', map_location=device)
    model = model_pr(nOut=256, n_class=5994 ,margin=0.2 , scale=30)

    # Proposed model
    # model_dict = torch.load('<< FAST_FT79.txt 경로 >>', map_location=device)
    # model = model(nOut=256, n_class=5994 ,margin=0.2 , scale=30)

    #######################################################################################################################################

    model.load_state_dict(model_dict['network'])

    model = model.to(device)

    model.eval()
    ## ---------------------------------------------------------------------- ##



  ## ================================================================= ##
  ## Extract speaker embeddings per subsegment
  fname2fpath = load_dict_txt(wav_scp)
  nl = linecount(reclist)
  for l, line in enumerate(open(reclist)):
    fname = line.rstrip().split(' ')[0]
    wpath = fname2fpath[fname]
    vpath = f'{vad_dir}/{fname}.lab'
    spath = f'{seg_dir}/{fname}.seg'
    epath = f'{emb_dir}/{fname}.ark'

    if not os.path.isfile(epath) or os.stat(epath).st_size == 0 or not os.path.isfile(spath) or os.stat(spath).st_size == 0:
      start_time = time.time()
      print(f'Processing {fname}... [{l+1}/{nl}]', end=" ")
      with open(spath, 'wt') as f_seg, open(epath, 'wb') as f_emb, torch.no_grad():
        ## Load VAD outputs
        vad = np.atleast_2d(
          (np.loadtxt(vpath, usecols=(0, 1))*fs).astype('int64')
        ) # seconds -> samples
        vad = np.atleast_2d(np.loadtxt(vpath, usecols=(0, 1))) # in seconds
        vad = vad[(vad[:,1] - vad[:,0]) > MIN_SEG_DUR] # remove too short segments
        vad = (vad * fs).astype(int) # in samples

        ## Process active segments
        for segnum in range(vad.shape[0]):
          ## Load segment
          seg, fs = sf.read(wpath, dtype='float32', 
            start=vad[segnum, 0], stop=vad[segnum, 1])

          ## Segment-level acoustic feature extraction
          seg = torch.from_numpy(seg[None]).to(device)
          
          ## ------------------------------------------- ##
          ## Extract embeddings from batched subsegments
          if batch_mode:
            ## Batch-process up to the (last-1)-th subsegment
            print('batch mode no!')
            exit()
          ## ------------------------------------------- ##
          ## Extract embeddings one by one (per subsegment)
          else:
            ## Subsegment-level speaker embedding extraction
            start, end = (0, segLen) # in frames
              ## Speaker embedding
            if seg.size()[-1] == 0:
                print('no emb::', seg)
            else:
              ################################################################################################################# 실습 #3
              emb = model.SpeakerNet.backbone.

              #################################################################################################################

              if emb is not None:
                for _emb in emb:
                    _emb = _emb.cpu().numpy()
                    subsegid = f'{fname}_{segnum:04}-{start:08}-{(start+segLen):08}'
                    st = round(vad[segnum,0]/fs + start*winSht/fs, 3)
                    et = round(st + segLen*winSht/fs, 3)
                    f_seg.write(f'{subsegid} {fname} {st} {et}\n')
                    write_kaldi_binary(f_emb, _emb, key=subsegid)
                    ## Update positions
                    start += segSht
                  
            ## ------------------------------------------- ##
      time_consumed = time.time() - start_time
      print(f'(Consumed {time_consumed:.2f} s)')
  ## End of for loop
  # exit()
  

  ## ================================================================= ##
  ## Perform clustering on subsegment-level embeddings
  num_spk = num_oracle_spks
  if num_spk is not None:
    rttm_dir += '_%dspk' % num_spk
    result_file += '_%dspk' % num_spk
    os.makedirs(rttm_dir, exist_ok=True)

  clst_types = cluster_type.split('+')

  nl = linecount(reclist)
  for l, line in enumerate(open(reclist)):
    fname = line.rstrip().split(' ')[0]
    spath = f'{seg_dir}/{fname}.seg'
    epath = f'{emb_dir}/{fname}.ark'
    rpath = f'{rttm_dir}/{fname}.rttm'

    if not os.path.isfile(rpath) or os.stat(rpath).st_size == 0 or args.overwrite:
      print(f'Processing {fname}... [{l+1}/{nl}]')
      sseg2time = load_dict_segments(spath)
      sseg2emb = {k:v for k,v in read_kaldi_arkscp(epath)}
      ssegids = sorted(sseg2emb)
      emb_mat = np.stack(tuple(sseg2emb[ssegid] for ssegid in ssegids))

      ## Compute affinity matrix
      assert score_type == "cos"
      
      emb_mat = np.squeeze(emb_mat, axis=1)
      emb_mat_T = np.transpose(emb_mat, axes=(1,0))
      sim_mat = compute_css_pairwise(emb_mat_T, emb_mat_T)


      ## Clustering algorithms
      if clst_types[0].lower() == 'ahc':
        ## Agglomerative hierarchical clustering
        thr, _ = twoGMMcalib_lin(sim_mat.ravel(), niters=20)
        # ----------------------------------------------------------------- #
        if fast_clustering:
          sim_mat = squareform(-sim_mat, checks=False)
          lin_mat = fastcluster.linkage(sim_mat, method='average', preserve_input='False')
          del sim_mat
          adjust = abs(lin_mat[:, 2].min())
          lin_mat[:, 2] += adjust
          labels = fcluster(lin_mat, -(thr+0.0) + adjust, criterion='distance') - 1
        # ----------------------------------------------------------------- #
        else:
          labels = AHC(sim_mat, threshold=thr)
        print(f'#clusters = {len(set(labels))}, AHC thres = {thr:.2f}' % thr)

      elif clst_types[0].lower() == 'spec':
        ## Spectral clustering
        if fast_clustering:
          clust_obj = diar.Spec_Clust_unorm(min_num_spkrs=1, max_num_spkrs=10)
          clust_obj.do_spec_clust(sim_mat=sim_mat, k_oracle=num_spk, p_val=p_val)
          labels = clust_obj.labels_

        else:
          print("no_clustering_options")
        #   labels = configs.icassp2018_clusterer.predict(emb_mat)

      ## Convert labels to speaker boundaries
      lol = []
      for sidx, ssegid in zip(labels, ssegids):
      # for sidx, ssegid, emb in zip(labels, ssegids, emb_mat):
        spkid = '%s_%d' % (fname, sidx)
        recid, st, et = sseg2time[ssegid]
        lol.append([recid, st, et, spkid])

      ## Merge adjacent sub-segments of the same speakers
      lol.sort(key=lambda x: float(x[1])) # sort in chronological order
      lol = diar.merge_ssegs_same_speaker(lol)

      ## Distribute duration of adjacent overlaps of different speakers
      lol_ovl = diar.distribute_overlap(lol)

      ## Write RTTM
      diar.write_rttm(lol_ovl, rpath)


  ## ================================================================= ##
  ## Evaluation
  sys_rttms_cat = rttm_dir+'.rttm'
  os.system(f'cat {rttm_dir}/*.rttm > {sys_rttms_cat}')

  scoring_command = f'python dscore/score.py '
  if ignore_overlaps:
    scoring_command += f'--ignore_overlaps '
  scoring_command += f'--collar {forgiveness_collar} \
    -r {ref_rttms_cat} -s {sys_rttms_cat} > {result_file}'
  os.system(scoring_command)



  ## ================================================================= ##
  ## Plot RTTMs for error analysis
  utt2err = {}
  for line in open(result_file):
    try:
      uttid, der, jer, *_ = line.split()
      utt2err[uttid] = tuple(map(float, (der, jer)))
    except:
      continue
  else:
    _, _, _, der, _, *_ = line.split()
    print(f'DER = {float(der):.2f}[%]')


