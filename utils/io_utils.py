import numpy as np
from scipy.io import wavfile
import soundfile as sf

import scipy.io as io
import hdf5storage as store

MAX_SIGNED_INT16 = 2**15 - 1

############################
###  Audio-IO functions  ###
############################

def read_aud(fpath, normalize=False):
  fpath = fpath.strip()
  ext = fpath.rsplit('.', 1)[-1]
  if ext=='raw' or ext=='pcm' or ext=='snd':
    aud = np.array(np.memmap(fpath, dtype='h', mode='r'))
  elif ext in ['wav']:
    _, aud = wavfile.read(fpath)
  elif ext in ['flac']:
    aud, _ = sf.read(fpath)
    aud = (aud * MAX_SIGNED_INT16).astype('int16')
  if normalize:
    aud = aud / float(MAX_SIGNED_INT16)
  return aud

def read_wav(fpath, normalize):
  ext = fpath.rsplit('.', 1)[-1]
  assert ext == 'wav'
  _, aud = wavfile.read(fpath)
  return aud

def write_wav(fpath, fs, data):
  # from scipy.io import wavfile as wavfile
  fpath = fpath.strip()
  if data.ndim == 1:
    data = data[:,None]
  n_chn = min(data.shape)
  if data.shape[0] == n_chn:
    data = data.T
  if data.max() < 1.0:
    data = data * float(MAX_SIGNED_INT16)
  wavfile.write(filename=fpath, rate=fs, data=data.astype('int16'))



###########################
###  File-IO functions  ###
###########################

def linecount(file):
  """ https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-\
      large-file-cheaply-in-python/68385697#68385697 """
  def _make_gen(reader):
    b = reader(2 ** 16)
    while b:
      yield b
      b = reader(2 ** 16)

  with open(file, "rb") as f:
    count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
  return count

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
  dic = dict()
  for line in open(filepath):
    key, val = line.rstrip().split(' ', 1)
    if dtype is not None:
      val = dtype(val)
    dic[key] = val
  return dic
