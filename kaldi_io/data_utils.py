import os
import numpy as np
from collections import OrderedDict as odict

import scipy.io as io
import hdf5storage as store

from kaldi_io.arkio import read_kaldi_binary
from kaldi_io.meta_utils import make_scp



def load_dict_segments(kaldi_segments_file):
  dic = {} # (s)segid2time
  for line in open(kaldi_segments_file):
    segid, recid, st, et = line.rstrip().split(' ')
    dic[segid] = (recid, float(st), float(et))
  return dic


def save_dict_npy(filepath, dictionary):
  np.save(filepath, dictionary)

def load_dict_npy(filepath):
  return np.load(filepath, allow_pickle=True).item()

def save_dict_mat(filepath, dictionary):
  io.savemat(filepath, mdict=dictionary)

def load_dict_mat(filepath):
  try:
    fdict = io.loadmat(filepath)
  except:
    fdict = store.loadmat(filepath)
  return fdict

def load_dict_txt(filepath, dtype=None):
  dic = odict()
  if isinstance(filepath, list):
    for file in filepath:
      dic.update(load_dict_txt(file, dtype))
    return dic
  elif isinstance(filepath, str):
    assert os.path.isfile(filepath), '%s does not exist!' % filepath
    for line in open(filepath):
      line = line.rstrip()
      if line != '':
        key, val = line.rstrip().split(' ', 1)
        if dtype is not None:
          val = dtype(val)
        dic[key] = val
    return dic

def load_dict_ark(arkfile):
  assert os.path.isfile(arkfile), '%s does not exist!!' % arkfile
  scpfile = arkfile.rsplit('.', 1)[0]+'.scp'
  if not os.path.isfile(scpfile):
    os.makedirs(os.path.dirname(scpfile), exist_ok=True)
    make_scp(arkfile, scp_out=scpfile)
  # assert os.path.isfile(scpfile), '%s does not exist!!' % scpfile
  return {key:read_kaldi_binary(arkp) for key, arkp in load_dict_txt(scpfile).items()}


def get_frmlen(wpath_or_nsamp, winLen, winSht, snip_edges=True):
  if isinstance(wpath_or_nsamp, str):
    wpath_or_nsamp = wavread.nsamples(wpath_or_nsamp.rstrip())
  assert isinstance(wpath_or_nsamp, int)

  if snip_edges:
    return int((wpath_or_nsamp - winLen) / winSht) + 1
  else:
    return int(np.round((wpath_or_nsamp + winSht/2) / winSht))


def rpad(mat23, padLen, time_axis=-2):
  """ reflection padding """
  frmLen = mat23.shape[time_axis]
  if frmLen < int(padLen/2):
    reps = np.ceil(padLen/2.0 / frmLen)
    shape = mat23.shape[:time_axis] + tuple([int(reps), 1])
    mat23 = np.tile(mat23, shape)
    frmLen = mat23.shape[time_axis]
  if frmLen <= padLen:
    res = padLen + 1 - frmLen
    pad = mat23[..., -res:, :]
    mat23 = np.concatenate([mat23, pad], axis=time_axis)
    if mat23.shape[time_axis] == padLen:
      mat23 = np.concatenate([mat23, mat23[..., -1:, :]], axis=time_axis)
  return mat23

def zpad(arr123, padLen, front=0, time_axis=-2):
  """ zero-padding """
  ndim = arr123.ndim
  assert ndim < 4
  if ndim == 1:
    time_axis = 0

  res = padLen - arr123.shape[time_axis]
  assert res >= 0
  if res or front:
    pad_width = [(0,0) for _ in range(ndim)]
    pad_width[time_axis] = (front, res)  # overwrite
    return np.pad(arr123, pad_width, mode='constant', constant_values=0)
  else:
    return arr123



def ThreadedGenerator(generator, num_cached=5):
  queue = Queue.Queue(maxsize=num_cached)
  sentinel = object()  # guaranteed unique reference

  ## Define producer
  def producer():
    for item in generator:
      queue.put(item)  # block=True
    queue.put(sentinel)

  ## Start producer (putting items into the queue)
  import threading
  thread = threading.Thread(target=producer)
  thread.daemon = True
  thread.start()

  ## Run as consumer (read items from queue, in the current thread)
  item = queue.get()
  while item is not sentinel:
    yield item
    queue.task_done()
    item = queue.get()  # block=True


