#!/usr/bin/env python3
import os, re
import time
import random
import fnmatch
import numpy as np
from numpy.random import permutation as permute
from collections import OrderedDict as o_dict
from tqdm import tqdm

from utils.io_utils import read_aud, save_dict_npy
from utils.wavread import nsamples

KALDI_UTILS = 'kaldi-utils'
kaldi_exec = dict()
kaldi_exec["wav-to-duration"] = '{}/wav-to-duration'.format(KALDI_UTILS)
# kaldi_exec["wav-reverberate"] = '{}/wav-reverberate'.format(KALDI_UTILS)



## Method for splitting the filelist
def split_scp(directory, filelist, nj):
  filename = filelist.split('/')[-1]
  ss = ''
  for j in range(nj):
    ss += '{}/{}.{}'.format(directory, filename, j+1)
    if j < nj-1:
      ss += ' '
  job_split = '{}/split_scp.pl {} {}'.format(KALDI_UTILS, filelist, ss)
  os.system(job_split)


def path2list(fpath, sort=False):
  flist = [line.rstrip() for line in open(fpath)]
  if sort:
    flist = sorted(flist)
  return flist

# def get_uttinfo(fpath):
#   spkID, uttID = fpath.split('/')[-2:]
#   uttID = spkID + '/' + uttID.rsplit('.', 1)[0]
#   return spkID, uttID

def get_uttinfo(fpath, level=2, delim='-', prepend=''):
  if level > 1:
    spkID, *uttID = fpath.rstrip().split('/')[-level:]
    spkID = prepend + spkID
    uttID = spkID+'/'+delim.join(uttID).rsplit('.', 1)[0]
  else:
    spkID, uttID = '', fpath.rstrip().split('/')[-level].rsplit('.', 1)[0]
  return spkID, uttID

def get_keys_ndarray(dic, ndim):
  return [k for k,v in dic.items() if isinstance(v, np.ndarray) and v.ndim >= ndim]


######################
###  file locater  ###
######################

def find_files(directory, pattern):
  for root, dirs, files in os.walk(directory):
    for basename in sorted(files):
      if fnmatch.fnmatch(basename, pattern):
        filename = os.path.join(root, basename)
        yield filename

def list_files(directory, pattern, sort=True):
  flist = []
  for fpath in find_files(directory, pattern):
    flist.append(fpath.rstrip())

  if sort:
    flist = list(np.sort(np.array(flist)))
  return flist


########################
###  flist handlers  ###
########################

def filter_flist(flist_in, flist_out, patterns, exclude=False):
  if not isinstance(patterns, (list, tuple)):
    patterns = [patterns]

  name_overlap = False
  if flist_in == flist_out:
    os.rename(flist_in, flist_in+'.tmp')
    flist_in += '.tmp'
    name_overlap = True

  with open(flist_out, 'wt') as f:
    if exclude:
      for fpath in open(flist_in):
        fpath = fpath.strip()
        for pattern in patterns:
          if re.search(pattern, fpath):
            break
        else:
          f.write(fpath+'\n')
    else:
      for fpath in open(flist_in):
        fpath = fpath.strip()
        for pattern in patterns:
          if re.search(pattern, fpath):
            f.write(fpath+'\n')
            break

  if name_overlap:
    os.remove(flist_in)

def rename_flist(flist_in, flist_out=None):
  if flist_out is None:
    raise ValueError('flist_out is not given !!')
  else:
    time.sleep(0.7)
    os.system('mv {0} {1}'.format(flist_in, flist_out))

def sort_flist(flist_in):
  os.system('mv {0} {0}.tmp'.format(flist_in))
  filelist = [ s.strip() for s in open(flist_in+'.tmp') ]
  filelist = list(np.sort(np.array(filelist)))
  with open(flist_in, 'wt') as f_flist:
    for fpath in filelist:
      f_flist.write(fpath+'\n')
  os.remove(flist_in+'.tmp')
  # os.system('rm {0}.tmp'.format(flist_in))

def concat_flists(flistlist, flist_out, sort=False):
  assert isinstance(flistlist, (list, tuple))
  # assert len(flistlist) > 1

  with open(flist_out, 'wt') as flist_new:
    for flist_in in flistlist:
      for fpath in open(flist_in):
        flist_new.write(fpath)

  if sort:
    sort_flist(flist=flist_new)
  # return flist_new

def concat_flines(flist_in, utt2voi=None, min_voi=500, n=5, sep='#', 
                  shuffle=True, ignore_res=True, flist_out=None):
  fpathlist = [ fpath.strip() for fpath in open(flist_in) ]
  n_files = len(fpathlist)
  if shuffle:
    rndord = permute(n_files)
    fpathlist = list(np.array(fpathlist)[rndord])

  with open(flist_out, 'wt') as newlist:
    if utt2voi is None:
      if ignore_res:
        fpathlist = fpathlist[:n_files-(n_files%n)]
        
      fline = []
      for i, fpath in enumerate(fpathlist):
        fline.append(fpath)
        if i % n == (n-1):
          fline = sep.join(fline)
          newlist.write(fline+'\n')
          fline = []
    else:
      spkIDlist = [fpath.split('/')[-2] for fpath in fpathlist]
      spk2dict = dict()
      for spkID in spkIDlist:
        spk2dict[spkID] = {'utts':[], 'dur':0}

      for fpath in fpathlist:
        fpath = fpath.strip()
        spkID, wavID = fpath.split('/')[-2:]
        uttID = "/".join([spkID, wavID.split('_')[0]])

        spk2dict[spkID]['utts'].append(fpath)
        spk2dict[spkID]['dur'] += utt2voi[uttID]
        if spk2dict[spkID]['dur'] > min_voi:
          # print(spk2dict[spkID]['dur'])
          fline = sep.join(spk2dict[spkID]['utts'])
          newlist.write(fline+'\n')
          spk2dict[spkID]['utts'] = []
          spk2dict[spkID]['dur'] = 0

def reduce_flist(flist, ups=20, shuffle=True, seed=1, flist_out=None):
  spk2fpath = o_dict()
  for fpath in open(flist):
    fpath = fpath.strip()
    spkID = fpath.split('/')[-2]
    if spkID in spk2fpath.keys():
      spk2fpath[spkID].append(fpath)
    else:
      spk2fpath[spkID] = [fpath]

  if seed is not None:
    # random.seed(seed)
    np.random.seed(seed)

  for spkID, fpathlist in spk2fpath.items():
    spk2fpath[spkID] = sorted(fpathlist)

  if shuffle:
    for spkID, fpathlist in spk2fpath.items():
      rndord = permute(len(fpathlist))
      spk2fpath[spkID] = list(np.array(fpathlist)[rndord])

  with open(flist_out, 'wt') as ff:
    for spkID, fpathlist in spk2fpath.items():
      for u in range(ups):
        ff.write(fpathlist[u]+'\n')


def reduce_wavscp(flist, ext='wav', ups=20, shuffle=True, seed=1, wavscp_out=None):
  # spk2utt = o_dict()
  spk2pipe = o_dict()
  for fpipe in open(flist):
    fpipe = fpipe.strip()
    spkID = fpipe.split('.'+ext)[0].split()[-1].split('/')[-2]
    if spkID in spk2pipe.keys():
      spk2pipe[spkID].append(fpipe)
    else:
      spk2pipe[spkID] = [fpipe]

  if seed is not None:
    random.seed(seed)

  with open(wavscp_out, 'wt') as ff:
    for spkID, fpipelist in spk2pipe.items():
      if shuffle:
        random.shuffle(fpipelist)

      for n, fpipe in enumerate(fpipelist):
        ff.write(fpipe+'\n')
        if n == ups-1:
          break




def make_flist(source_dir, extension='wav', rep_dict={}, ext_dict={}, 
               flist=None, sort=True):
  assert flist is not None
  if not isinstance(source_dir, (list, tuple)):
    source_dir = [source_dir]
  if not isinstance(extension, (list, tuple)):
    extension = [extension]

  os.makedirs(os.path.dirname(flist), exist_ok=True)
  with open(flist, 'wt') as f_flist:
    for src_dir in source_dir:
      if not os.path.exists(src_dir):
        raise Exception('{} does not exists !!'.format(src_dir))

      for ext in extension:
        for fpath in find_files(src_dir, '*.'+ext):
          ## replace filepath if necessary
          for before, after in list(rep_dict.items()):
            fpath = fpath.replace(before, after)
          ## replace extention if necessary
          for before, after in list(ext_dict.items()):
            fpath = fpath.replace(before, after)
          f_flist.write(fpath+'\n')
  if sort: sort_flist(flist)


def make_wavscp(source_dir, source_fs=16000, extension='wav', target_fs=16000, 
                dir_level=2, delim='-', wavscp=None, sort=True, 
                enc_and_bit='-e signed -b 16'):
  print('Make wav.scp for {}...'.format(source_dir))

  if not isinstance(source_dir, list):
    source_dir = [source_dir]

  if not isinstance(extension, list):
    extension = [extension]

  if wavscp is None:
    wavscp = source_dir + '/wav.scp'

  f_wavscp = open(wavscp, 'wt')
  for src_dir in source_dir:
    print('\t'+src_dir)
    if not os.path.exists(src_dir):
      raise Exception('%s does not exists !!' % src_dir)

    # f_utt2spk = open(src_dir + '/utt2spk', 'wt')
    for ext in extension:
      for fpath in find_files(src_dir, '*.'+ext):
        # uttID = "/".join(fpath.replace('.'+ext,'').split('/')[-2:])
        _, uttID = get_uttinfo(fpath, level=dir_level, delim=delim)
        # spkID = fpath.replace('.'+ext,'').split('/')[-2]  # only for speech
        # f_utt2spk.write('{0} {0}\n'.format(uttID))        # only for speech
        # f_utt2spk.write('{0} {0}\n'.format(uttID))  # only for noise (if needed)
        if source_fs is not None:
          if ext in ['wav', 'flac']:
            ment = '{0} sox -t {1} {2} -r {3} -t wav - |\n' \
                    .format(uttID, ext, fpath, int(target_fs))
          elif ext in ['raw', 'pcm', 'snd']:
            ment = '{0} sox -r {1} {5} -t {2} {3} -r {4} -t wav - |\n' \
                    .format(uttID, int(source_fs), ext, fpath, int(target_fs), 
                            enc_and_bit)
        else:
          assert ext == 'wav'
          ment = '{0} sox -t wav {1} -r {2} -t wav - |\n' \
                  .format(uttID, fpath, int(target_fs))
        f_wavscp.write(ment)
  f_wavscp.close()
  # f_utt2spk.close()

  if sort:
    sort_flist(wavscp)

def make_wavscp2(source_dir, extension='wav', dir_level=2, delim='-', 
                 wavscp=None, sort=True):
  print('Make wav.scp for {}...'.format(source_dir))

  if not isinstance(source_dir, list):
    source_dir = [source_dir]

  if not isinstance(extension, list):
    extension = [extension]

  if wavscp is None:
    wavscp = source_dir + '/wav.scp'
  if len(wavscp.split('/')) == 1:
    wavscp = './'+wavscp

  os.makedirs(os.path.dirname(wavscp), exist_ok=True)
  f_wavscp = open(wavscp, 'wt')
  for src_dir in source_dir:
    print('\t'+src_dir)
    if not os.path.exists(src_dir):
      raise Exception('%s does not exists !!' % src_dir)

    for ext in extension:
      for fpath in find_files(src_dir, '*.'+ext):
        _, uttID = get_uttinfo(fpath, level=dir_level, delim=delim)
        f_wavscp.write('{0} {1}\n'.format(uttID, fpath))
  f_wavscp.close()
  if sort:
    sort_flist(wavscp)


def write_to_reco2dur(src_dir, f_opened, extension, fs):
  for ext in extension:
    for fpath in find_files(src_dir, '*.'+ext):
      uttID = "/".join(fpath.replace('.'+ext,'').split('/')[-2:])
      aud = read_aud(fpath)
      ment = '{} {}\n'.format(uttID, aud.shape[0]/float(fs))
      f_opened.write(ment)
      # return ment

def make_reco2dur(wavlist=None, source_dir=None, fs=16000, extension='wav', 
                  reco2dur=None, sort=True):

  if wavlist is None and source_dir is not None:
    if not isinstance(source_dir, list):
      source_dir = [source_dir]
    wavlist = []
    for src_dir in source_dir:
      wavlist += list_files(src_dir, '*.'+extension)
    with open(reco2dur, 'wt') as f:
      for wpath in tqdm(wavlist):
        wpath = wpath.rstrip()
        uttID = "/".join(wpath.replace('.'+extension, '').split('/')[-2:])
        if extension == 'wav':
          nsamp = nsamples(wpath)
        else:
          nsamp = read_aud(wpath).shape[0]
        f.write('{} {}\n'.format(uttID, nsamp/float(fs)))

  elif wavlist is not None and source_dir is None:
    if not isinstance(wavlist, list):
      wavlist = [wavlist]
    with open(reco2dur, 'wt') as f:
      for _wavlist in wavlist:
        for wpath in tqdm(open(_wavlist)):
          wpath = wpath.rstrip()
          uttID = "/".join(wpath.replace('.'+extension, '').split('/')[-2:])
          if extension == 'wav':
            nsamp = nsamples(wpath)
          else:
            nsamp = read_aud(wpath).shape[0]
          f.write('{} {}\n'.format(uttID, nsamp/float(fs)))

  if sort:
    sort_flist(reco2dur)


def make_wavlist_from_commands(commands, wavlist):
  with open(wavlist, 'wt') as ff:
    for line in open(commands):
      wpath = line.rstrip().split()[-1]
      ff.write(wpath+'\n')


def prepend_path(filelist, prefix, out, rank=0, delim=" "):
  """ *Assume that each line of the filelist ends with the <filepath>.
      Just prepend the <prefix> to the beginning of the <filepath>. """
  if out == filelist:
    os.system('mv {0} {0}.tmp'.format(filelist))
    filelist += '.tmp'

  with open(out, 'wt') as f:
    for line in open(filelist):
      line = line.strip()
      if rank >= 0:
        line_split = line.split(delim)
        line = delim.join(line_split[:rank] + [prefix+line_split[rank]] + line_split[rank+1:])
      elif rank == -1:
        line = delim.join([prefix+ll for ll in line.split(delim)])
      f.write(line+'\n')

def prepend_uttids(scpfile, prefix):
  os.system('mv {0} {0}.tmp'.format(scpfile))
  scpfile_tmp = scpfile + '.tmp'
  with open(scpfile, 'wt') as f:
    for line in open(scpfile_tmp):
      uttID, rest = line.rstrip().split(' ', 1)
      f.write('{}{} {}\n'.format(prefix, uttID, rest))

def append_uttids(scpfile, postfix):
  os.system('mv {0} {0}.tmp'.format(scpfile))
  scpfile_tmp = scpfile + '.tmp'
  with open(scpfile, 'wt') as f:
    for line in open(scpfile_tmp):
      uttID, rest = line.rstrip().split(' ', 1)
      f.write('{}{} {}\n'.format(uttID, postfix, rest))

def replace_uttids(scpfile, rep_dict):
  os.system('mv {0} {0}.tmp'.format(scpfile))
  scpfile_tmp = scpfile + '.tmp'
  with open(scpfile, 'wt') as f:
    for line in open(scpfile_tmp):
      uttID, rest = line.rstrip().split(' ', 1)
      for bef, aft in rep_dict.items():
        uttID = uttID.replace(bef, aft)
      f.write('{} {}\n'.format(uttID, rest))
  os.remove(scpfile_tmp)


def bipart_flist(flist_in, portion=2/3, shuffle=False):
  assert portion > 0.5
  # dirname = os.path.dirname(flist_in)
  # basename = os.path.basename(flist_in)

  fpathlist = [fpath.rstrip() for fpath in open(flist_in)]
  n_files = len(fpathlist)
  if shuffle:
    rndord = permute(n_files)
    fpathlist = list(np.array(fpathlist)[rndord])

  n_part1 = int(n_files * portion)
  # n_part2 = n_files - n_part1

  flist_out1 = flist_in + '_tr'
  flist_out2 = flist_in + '_cv'

  with open(flist_out1, 'wt') as ff:
    for fpath in fpathlist[:n_part1]:
      ff.write(fpath+'\n')

  with open(flist_out2, 'wt') as ff:
    for fpath in fpathlist[n_part1:]:
      ff.write(fpath+'\n')


def make_utt2len(flist_in, winLen=512, winSht=128, snip_edges=True, 
                 utt2len_path='utt2len'):
  with open(utt2len_path, 'wt') as f:
    for fpath in tqdm(open(flist_in)):
      fpath = fpath.rstrip()
      spkID, uttID = fpath.split('/')[-2:]
      uttID = spkID + '/' + uttID.split('.')[0]

      ## Total number of frames
      n_samp = max(read_aud(fpath).shape)
      if snip_edges:
        totfrm = int((n_samp - winLen) / winSht) + 1
      else:
        totfrm = int(np.round((n_samp + winSht/2) / winSht))
      f.write('{} {}\n'.format(uttID, totfrm))


def make_utt2vad(flist_in, vad_opts, utt2vad_path='utt2vad.npy'):
  from utils.feat_utils import probe_vad, kaldi_vad

  utt2vad = dict()
  for fpath in tqdm(open(flist_in)):
    # probe_vad(fpath, **vad_opts); exit()
    fpath = fpath.rstrip()
    _, uttID = get_uttinfo(fpath)
    utt2vad[uttID] = kaldi_vad(fpath, **vad_opts)
  save_dict_npy(utt2vad_path, utt2vad)
