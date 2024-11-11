from __future__ import division, print_function, absolute_import

import sys
import numpy
import struct
import warnings


class WavFileWarning(UserWarning):
  pass

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xfffe
KNOWN_WAVE_FORMATS = (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT)


# assumes file pointer is immediately after the 'fmt ' id
def _read_fmt_chunk(fid, endian):
  res = struct.unpack(endian+'iHHIIHH',fid.read(20))
  size, comp, noc, rate, sbytes, ba, bits = res
  if comp not in KNOWN_WAVE_FORMATS or size > 16:
    comp = WAVE_FORMAT_PCM
    warnings.warn("Unknown wave file format", WavFileWarning)
    if size > 16:
      fid.read(size - 16)

  return size, comp, noc, rate, sbytes, ba, bits


# assumes file pointer is immediately after the 'data' id
def _read_data_chunk(fid, comp, noc, bits, ba, start=0, dur=0, 
                     mmap=False, endian='<'):
  size = struct.unpack(endian+'i', fid.read(4))[0]
  # assert fid.tell() == 44
  size = size - ba*start if dur == 0 else ba*dur
  fid.seek(ba*start, 1)

  bytes = bits//8
  if bits == 8:
    dtype = 'u1'
  else:
    dtype = endian+'i%d' % bytes if comp==1 else endian+'f%d' % bytes

  if not mmap:
    data = numpy.fromstring(fid.read(size), dtype=dtype)
  else:
    start = fid.tell()
    data = numpy.memmap(fid, dtype=dtype, mode='c', offset=start,
                        shape=(size//bytes,))
    fid.seek(start + size)

  if noc > 1:
    return data.reshape(-1, noc)
  return data


def _skip_unknown_chunk(fid, endian):
  data = fid.read(4)
  size = struct.unpack(endian+'i', data)[0]
  fid.seek(size, 1)
  return size


def _read_riff_chunk(fid):
  endian = '<'
  str1 = fid.read(4)
  if str1 == b'RIFX':
    endian = '>'
  elif str1 != b'RIFF':
    raise ValueError("Not a WAV file.")

  fsize = struct.unpack(endian+'I', fid.read(4))[0] + 8 # 8 for Chunk ID & Chunk Size
  str2 = fid.read(4)
  if (str2 != b'WAVE'):
    raise ValueError("Not a WAV file.")
  if str1 == b'RIFX':
    endian = '>'
  return fsize, endian



# def read_header(fname, mmap=False):
#   fid = fname if hasattr(fname,'read') else open(fname, 'rb')
#   try:
#     fsize, endian = _read_riff_chunk(fid)  # file size in bytes
#     noc = 1
#     bits = 8
#     comp = WAVE_FORMAT_PCM

#     while (fid.tell() < fsize):
#       ## read the next chunk
#       chunk_id = fid.read(4)
#       if chunk_id == b'fmt ':
#         size, comp, noc, rate, sbytes, ba, bits = _read_fmt_chunk(fid, endian)
#         # assert fid.tell() == 36
#         # if dur:
#         #   assert fsize >= 44 + ba * (start+dur)
#         #   fsize = 44 + ba * (start+dur)

#       elif chunk_id == b'fact':
#         _skip_unknown_chunk(fid)
#       elif chunk_id == b'data':
#         # assert fid.tell() == 40
#         size = struct.unpack(endian+'i', fid.read(4))[0]
#         # assert fid.tell() == 44
#         return rate, noc, int(size/ba), comp, bits, ba
#         # data = _read_data_chunk(fid, comp, noc, bits, ba, 
#         #                         start, dur, mmap=mmap)
#       elif chunk_id == b'LIST':
#         ## Someday this could be handled properly but for now skip it
#         _skip_unknown_chunk(fid)
#       else:
#         warnings.warn("Chunk (non-data) not understood, skipping it.", WavFileWarning)
#         _skip_unknown_chunk(fid)
#   finally:
#     if not hasattr(fname,'read'):
#       fid.close()
#     else:
#       fid.seek(0)

#   return rate, data


def nchannels(fname, endian='<'):
  fid = open(fname, 'rb')
  try:
    fsize, endian = _read_riff_chunk(fid)  # file size in bytes
    while fid.tell() < 40:
      ## read the next chunk
      chunk_id = fid.read(4)
      if chunk_id == b'fmt ':
        # fid.read(6)  # 4 for "size", 2 for "comp"
        # noc = struct.unpack(endian+'H',fid.read(2))
        size, comp, noc, rate, sbytes, ba, bits = _read_fmt_chunk(fid, endian)
        break
      elif chunk_id == b'fact':
        _skip_unknown_chunk(fid, endian)
      elif chunk_id == b'LIST':
        ## Someday this could be handled properly but for now skip it
        _skip_unknown_chunk(fid, endian)
      else:
        warnings.warn("Chunk (non-data) not understood, skipping it.", WavFileWarning)
        _skip_unknown_chunk(fid, endian)
  finally:
    fid.close()
  return noc


def nsamples(fname):
  fid = open(fname, 'rb')
  try:
    fsize, endian = _read_riff_chunk(fid)  # file size in bytes
    while fid.tell() < 44:
      ## read the next chunk
      chunk_id = fid.read(4)
      if chunk_id == b'fmt ':
        size, comp, noc, rate, sbytes, ba, bits = _read_fmt_chunk(fid, endian)
      elif chunk_id == b'data':
        size = struct.unpack(endian+'i', fid.read(4))[0]
        # assert fid.tell() == 44
        break
      elif chunk_id == b'fact':
        _skip_unknown_chunk(fid, endian)
      elif chunk_id == b'LIST':
        ## Someday this could be handled properly but for now skip it
        _size = _skip_unknown_chunk(fid, endian)
        size = fsize - size - _size - 36
      else:
        warnings.warn("Chunk (non-data) not understood, skipping it.", WavFileWarning)
        _skip_unknown_chunk(fid, endian)
  finally:
    fid.close()
  return int(size/ba)

def srate(fname, mmap=False):
  fid = open(fname, 'rb')
  try:
    fsize, endian = _read_riff_chunk(fid)  # file size in bytes
    while (fid.tell() < fsize):
      ## read the next chunk
      chunk_id = fid.read(4)
      if chunk_id == b'fmt ':
        size, comp, noc, rate, sbytes, ba, bits = _read_fmt_chunk(fid, endian)
        fid.close()
        return rate
      elif chunk_id == b'fact':
        _skip_unknown_chunk(fid, endian)
      elif chunk_id == b'data':
        raise NotImplementedError("Something's wrong; this part should not be reached.")
      elif chunk_id == b'LIST':
        ## Someday this could be handled properly but for now skip it
        _skip_unknown_chunk(fid, endian)
      else:
        warnings.warn("Chunk (non-data) not understood, skipping it.", WavFileWarning)
        _skip_unknown_chunk(fid, endian)
  finally:
    fid.close()

# def srate(fname, mmap=False):
#   if hasattr(fname,'read'):
#     fid = fname
#     mmap = False
#   else:
#     fid = open(fname, 'rb')

#   try:
#     fsize, endian = _read_riff_chunk(fid)  # file size in bytes
#     noc = 1
#     bits = 8
#     comp = WAVE_FORMAT_PCM


#     # chunk_id = fid.read(4)
#     # assert chunk_id == b'fmt '
#     # size, comp, noc, rate, sbytes, ba, bits = _read_fmt_chunk(fid)

#     # chunk_id = fid.read(4)
#     # assert chunk_id == b'data'
#     # data = _read_data_chunk(fid, comp, noc, bits, ba, 
#     #                         start, dur, mmap=mmap)


#     while (fid.tell() < fsize):
#       ## read the next chunk
#       chunk_id = fid.read(4)
#       if chunk_id == b'fmt ':
#         size, comp, noc, rate, sbytes, ba, bits = _read_fmt_chunk(fid, endian)
#         return rate
#       elif chunk_id == b'fact':
#         _skip_unknown_chunk(fid, endian)
#       elif chunk_id == b'data':
#         raise NotImplementedError("Something's wrong; this part should not be reached.")
#       elif chunk_id == b'LIST':
#         ## Someday this could be handled properly but for now skip it
#         _skip_unknown_chunk(fid, endian)
#       else:
#         warnings.warn("Chunk (non-data) not understood, skipping it.", WavFileWarning)
#         _skip_unknown_chunk(fid, endian)
#   finally:
#     if not hasattr(fname,'read'):
#       fid.close()
#     else:
#       fid.seek(0)


def read(fname, start=0, dur=0, normalize=False, dtype='float32', mmap=False):
  if hasattr(fname,'read'):
    fid = fname
    mmap = False
  else:
    fid = open(fname, 'rb')

  try:
    fsize, endian = _read_riff_chunk(fid)  # file size in bytes
    noc = 1
    bits = 8
    comp = WAVE_FORMAT_PCM


    # chunk_id = fid.read(4)
    # assert chunk_id == b'fmt '
    # size, comp, noc, rate, sbytes, ba, bits = _read_fmt_chunk(fid)

    # chunk_id = fid.read(4)
    # assert chunk_id == b'data'
    # data = _read_data_chunk(fid, comp, noc, bits, ba, 
    #                         start, dur, mmap=mmap)


    while (fid.tell() < fsize):
      ## read the next chunk
      chunk_id = fid.read(4)
      if chunk_id == b'fmt ':
        size, comp, noc, rate, sbytes, ba, bits = _read_fmt_chunk(fid, endian)
        # assert fid.tell() == 36
        if dur:
          assert fsize >= 44 + ba * (start+dur)
          fsize = 44 + ba * (start+dur)

      elif chunk_id == b'fact':
        _skip_unknown_chunk(fid, endian)
      elif chunk_id == b'data':
        # assert fid.tell() == 40
        data = _read_data_chunk(fid, comp, noc, bits, ba, 
                                start, dur, mmap=mmap, endian=endian)
      elif chunk_id == b'LIST':
        ## Someday this could be handled properly but for now skip it
        _skip_unknown_chunk(fid, endian)
      else:
        warnings.warn("Chunk (non-data) not understood, skipping it.", WavFileWarning)
        _skip_unknown_chunk(fid, endian)
  finally:
    if not hasattr(fname,'read'):
      fid.close()
    else:
      fid.seek(0)

  if normalize:
    data = (data / float(2**(bits-1))).astype(dtype)
  return rate, data


