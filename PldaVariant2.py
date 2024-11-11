import os
import numpy as np
from functools import reduce


def save_dict_npy(filepath, dictionary):
  np.save(filepath, dictionary)

def load_dict_npy(filepath):
  return np.load(filepath, allow_pickle=True).item()


def dict2list(dictionary, sort=True):
  """ values of dictionary in shape (ivDim, num_utts) """
  keys = sorted(list(dictionary.keys()))
  data = [dictionary[key] for key in keys]
  if sort:
    data.sort(key=lambda x: x.shape[1]) # sort by num_utts
  return data


def cosine_similarity(A, B, axis=0):
  numer = (A*B).sum(axis=axis)
  denom1 = np.sqrt((A**2).sum(axis=axis))
  denom2 = np.sqrt((B**2).sum(axis=axis))
  return numer / (denom1 * denom2)

def compute_css_pairwise(enr, tst):
  """ enr/tst must be in shape (ivDim, n_samples) """
  assert enr.shape == tst.shape
  enr = length_norm(enr)
  tst = length_norm(tst)
  return enr.T.dot(tst)


def length_norm(ivec_data, SCALE_BY_DIM=0):
  if isinstance(ivec_data, np.ndarray):
    """ ivec_mat.shape == (ivDim, n_samples) """
    scale = (1 if not SCALE_BY_DIM else np.sqrt(ivec_data.shape[0]))
    ivec_norm = np.linalg.norm(ivec_data, ord=2, axis=0, keepdims=True)
    return ivec_data / (scale * ivec_norm)
  elif isinstance(ivec_data, list):
    return [length_norm(spk_data, SCALE_BY_DIM) for spk_data in ivec_data]
  elif isinstance(ivec_data, dict):
    return {spkID:length_norm(spk_data, SCALE_BY_DIM) 
            for spkID, spk_data in ivec_data.items()}


class PldaVariant(object):
  def __init__(self, model_type='jb', dataDim=0, LDAdim=0, Vdim=0, Udim=0, 
               do_centering=False, do_WCCN=True, do_whitening=True, do_lengthnorm=True, 
               do_mindiv=False):
    self.model_type = model_type

    self.n_clas = 0
    self.rawDim = dataDim # for computing score (i.e. LLR)
    self.dataDim = dataDim

    self.m0 = np.zeros((self.rawDim, 1), dtype='float32')
    self.LDAmat = np.eye(self.rawDim, dtype='float32')
    self.B = np.eye(self.dataDim, dtype='float32')
    self.W = np.eye(self.dataDim, dtype='float32')

    self.do_centering = do_centering
    self.LDAdim = LDAdim
    self.do_WCCN = do_WCCN
    self.do_whitening = do_whitening
    self.m1 = 0  # maintained as 0 if "do_whitening" is False
    self.do_lengthnorm = do_lengthnorm

    self.Vdim = Vdim
    self.Udim = Udim
    self.mu = 0  # maintained as 0 if "model_type" != 'tcov'
    self.V = 0
    self.U = 0
    self.S_mu = 0
    self.S_eps = 0
    self.Sigma = 0
    self.do_md_step = do_mindiv # minimum-divergence training

    self.SIGMA_INIT_METHOD = 'covariance'  # 'covariance', 'random'
    self.SIGMA_SCALE = 1.  # 0.5 or 1.0
    # self.preprocess_done = False

    self.Gamma = None
    self.Lambda_tot = None
    self.Sigma_bc = None
    self.Sigma_wc = None

    self.ext = 'npy'

  #### ========================================================================= ####

  def save_model(self, model_path):
    lgm = {
      'model_type':self.model_type,
      'rawDim':self.rawDim,
      'dataDim':self.dataDim,
      'do_centering':self.do_centering,
      'm0':self.m0,
      'LDAdim':self.LDAdim,
      'LDAmat':self.LDAmat,
      'do_WCCN':self.do_WCCN,
      'B':self.B,
      'do_whitening':self.do_whitening,
      'm1':self.m1,
      'W':self.W,
      'mu':self.mu,
      'do_lengthnorm':self.do_lengthnorm,
      'V':self.V,
      'U':self.U,
      'Sigma':self.Sigma,
      'S_mu':self.S_mu,
      'S_eps':self.S_eps,
      'Gamma':self.Gamma,
      'Lambda_tot':self.Lambda_tot,
      'Sigma_bc':self.Sigma_bc,
      'Sigma_wc':self.Sigma_wc,
    }

    if not model_path.endswith('.npy'):
      model_path += '.npy'
    save_dict_npy(model_path, lgm)
    print('Saved \"{}\" model to {}...'.format(self.model_type, model_path))

  def load_model(self, model_path, return_dict=False):
    assert os.path.isfile(model_path), "%s does not exist!!" % model_path
    lgm = load_dict_npy(model_path)

    if return_dict:
      return lgm
    else:
      self.model_type = lgm['model_type']
      self.rawDim = lgm['rawDim']
      self.dataDim = lgm['dataDim']
      self.do_centering = lgm['do_centering']
      self.m0 = lgm['m0']
      self.LDAdim = lgm['LDAdim']
      self.LDAmat = lgm['LDAmat']
      self.do_WCCN = lgm['do_WCCN']
      self.B = lgm['B']
      self.do_whitening = lgm['do_whitening']
      self.m1 = lgm['m1']
      self.W = lgm['W']
      self.mu = lgm['mu']
      self.do_lengthnorm = lgm['do_lengthnorm']
      self.V = lgm['V']
      self.U = lgm['U']
      self.Sigma = lgm['Sigma']
      self.S_mu = lgm['S_mu']
      self.S_eps = lgm['S_eps']
      self.Gamma = lgm['Gamma']
      self.Lambda_tot = lgm['Lambda_tot']
      self.Sigma_bc = lgm['Sigma_bc']
      self.Sigma_wc = lgm['Sigma_wc']
      del lgm

  def print_model(self):
    def print_shape_if_nonzero(template, arr):
      is_integer = isinstance(arr, int)
      nonzero = arr if is_integer else arr.sum()
      value = nonzero if is_integer else (arr.shape if nonzero else nonzero)
      print(template.format(value))

    print('\n++++++++++++++++++++')
    print("+ model_type: \'{}\'\n+".format(self.model_type))
    print_shape_if_nonzero('+ rawDim: {}', self.rawDim)
    print_shape_if_nonzero('+ dataDim: {}', self.dataDim)
    print_shape_if_nonzero('+ m0: {}\n+', self.m0)
    print_shape_if_nonzero('+ LDAdim: {}', self.LDAdim)
    print_shape_if_nonzero('+ LDAmat: {}\n+', self.LDAmat.T)
    print_shape_if_nonzero('+ B: {}\n+', self.B.T)
    print_shape_if_nonzero('+ m1: {}', self.m1)
    print_shape_if_nonzero('+ W: {}\n+', self.W)
    print_shape_if_nonzero('+ mu: {}', self.mu)
    print_shape_if_nonzero('+ V: {}', self.V)
    print_shape_if_nonzero('+ U: {}', self.U)
    print_shape_if_nonzero('+ Sigma: {}', self.Sigma)
    print_shape_if_nonzero('+ S_mu: {}', self.S_mu)
    print_shape_if_nonzero('+ S_eps: {}', self.S_eps)
    print('++++++++++++++++++++\n')


  #### ========================================================================= ####

  def compute_llk(self, pathlist):
    ## Total covariance matrix for the model with integrated out latent variables
    if self.model_type in ['simp', 'std']:
      Sigma_tot = np.dot(self.V, self.V.T) + np.dot(self.U, self.U.T) + self.Sigma
    elif self.model_type in ['tcov', 'jb']:
      Sigma_tot = self.S_mu + self.S_eps
    Sigma_tot_inv = np.linalg.inv(Sigma_tot)

    ## Compute log-determinant of the Sigma_tot
    evals, _ = np.linalg.eig(Sigma_tot)
    log_det = np.sum(np.log(evals))  # product of eigvals == determinant

    const = -0.5 * (self.dataDim*np.log(2*np.pi) + log_det)
    llk_m, n_dvecs = (0.0, 0)
    for dvec_path in pathlist:
      ## Centering + LDA + WCCN + Whitening #@!
      dvec_dict = self.preprocess_steps(load_dict_npy(dvec_path))
      dvec_stack = np.hstack(dict2list(dvec_dict))
      llk_m += -0.5 * np.sum(np.dot(dvec_stack.T, Sigma_tot_inv) * dvec_stack.T)
      n_dvecs += dvec_stack.shape[1]
    llk_m /= float(n_dvecs)
    return llk_m + const

  def compute_Gamma_Lambda_tot(self):
    inv = np.linalg.inv
    ### Full-PLDA scoring method ####
    if self.model_type in ['simp', 'std']:
      Sigma_bc = np.dot(self.V, self.V.T)
      Sigma_wc = np.dot(self.U, self.U.T) + self.Sigma
      Sigma_tot = Sigma_bc + Sigma_wc
      self.Lambda_tot = -inv(Sigma_wc + 2*Sigma_bc) + inv(Sigma_wc)
      self.Gamma = -inv(Sigma_wc + 2*Sigma_bc) - inv(Sigma_wc) + 2*inv(Sigma_tot)

    elif self.model_type in ['tcov']:
      Sigma_bc = self.S_mu
      Sigma_wc = self.S_eps
      Sigma_tot = Sigma_bc + Sigma_wc
      self.Lambda_tot = -inv(Sigma_wc + 2*Sigma_bc) + inv(Sigma_wc)
      self.Gamma = -inv(Sigma_wc + 2*Sigma_bc) - inv(Sigma_wc) + 2*inv(Sigma_tot)

    elif self.model_type in ['jb']:
      Sigma_bc = self.S_mu
      Sigma_wc = self.S_eps
      Sigma_tot = Sigma_bc + Sigma_wc
      Sigma_tot_inv = inv(Sigma_tot)
      self.Lambda_tot = reduce(np.dot, 
        [inv(2*Sigma_bc + Sigma_wc), Sigma_bc, inv(Sigma_wc)]
      )
      self.Gamma = Sigma_tot_inv - inv(
        Sigma_tot - reduce(np.dot, [Sigma_bc, Sigma_tot_inv, Sigma_bc])
      )
    self.Sigma_bc = Sigma_bc
    self.Sigma_wc = Sigma_wc

  def map_estimate(self, vmat, preprocess_done=False):
    if not preprocess_done:
      vmat = self.preprocess_steps(vmat)
    Sigma_bc, Sigma_wc = self.Sigma_bc, self.Sigma_wc
    Sigma_wc_inv = np.linalg.inv(Sigma_wc)
    trans = np.linalg.inv(
      np.eye(vmat.shape[0], dtype=vmat.dtype) + 
      reduce(np.dot, [Sigma_bc.T, Sigma_wc_inv, Sigma_bc])
    )
    trans = trans.dot(Sigma_bc.T).dot(Sigma_wc_inv)
    # return trans.dot(vmat)
    return trans.T.dot(vmat)

  def compute_llkr(self, vmat_enr, vmat_tst, tpb=100000, left_done=False, right_done=False):
    assert vmat_enr.shape == vmat_tst.shape
    score_arr = []
    n_batches = int(np.ceil(vmat_enr.shape[1] / float(tpb)))
    for i in range(n_batches):
      enr = vmat_enr[:,i*tpb:(i+1)*tpb]
      tst = vmat_tst[:,i*tpb:(i+1)*tpb]
      if not left_done:
        enr = self.preprocess_steps(enr)
      if not right_done:
        tst = self.preprocess_steps(tst)

      Gamma11 = np.sum(np.dot(enr.T, self.Gamma) * enr.T, axis=1)
      Gamma22 = np.sum(np.dot(tst.T, self.Gamma) * tst.T, axis=1)
      LLR = 2 * np.sum(np.dot(enr.T, self.Lambda_tot) * tst.T, axis=1)
      score_arr.append(Gamma11 + Gamma22 + LLR)
    return np.hstack(score_arr)

  def compute_llkr_pairwise(self, enr, tst, left_done=False, right_done=False):
    assert enr.shape == tst.shape
    if not left_done:
      enr = self.preprocess_steps(enr)
    if not right_done:
      tst = self.preprocess_steps(tst)

    Gamma11 = enr.T.dot(self.Gamma).dot(enr)
    Gamma22 = tst.T.dot(self.Gamma).dot(tst)
    LLR = 2 * enr.T.dot(self.Lambda_tot).dot(tst)
    return Gamma11 + Gamma22 + LLR

  def compute_css(self, vmat_enr, vmat_tst, tpb=200000, left_done=False, right_done=False):
    """ LDAdim = 500, (Cen, B, W, LN) = (T, F, F, T) """
    assert vmat_enr.shape == vmat_tst.shape
    score_arr = []
    n_batches = int(np.ceil(vmat_enr.shape[1] / float(tpb)))
    for i in range(n_batches):
      enr = vmat_enr[:,i*tpb:(i+1)*tpb]
      tst = vmat_tst[:,i*tpb:(i+1)*tpb]
      if not left_done:
        enr = self.preprocess_steps(enr)
      if not right_done:
        tst = self.preprocess_steps(tst)
      score_arr.append(cosine_similarity(enr, tst, axis=0))
    return np.hstack(score_arr)


  #### ========================================================================= ####

  def preprocess_steps(self, pathlist, training=False):
    """ pathlist is a list of *.ark files containing speaker embeddings """

    if training:
      ## Check file existance
      if not isinstance(pathlist, list):
        assert isinstance(pathlist, str)
        pathlist = [pathlist]
      for path in pathlist:
        assert os.path.isfile(path), "%s does not exist!" % path
      ## Sort paths by name
      pathlist.sort()

      ### 0) Count the number of classes
      ### *Note: the dictionary must have been constructed as follows:
      ### dic = {spkid1:dvec_arr1<N1,dim>, spkid2:dvec_arr2<N2,dim>, ...}
      for dvec_path in pathlist:
        self.n_clas += len(load_dict_npy(dvec_path))
      print('#classes = {}'.format(self.n_clas))


      ### 1) Centering ###
      if self.do_centering:
        print('\tCentering training data...')
        n_dvecs = 0
        for dvec_path in pathlist:
          dvec_stack = np.hstack(dict2list(load_dict_npy(dvec_path)))
          n_dvecs += dvec_stack.shape[1]
          self.m0 += np.sum(dvec_stack, axis=1, keepdims=True)
        self.m0 /= n_dvecs


      ### 2) LDA ###
      if self.LDAdim:
        # print('\ttraining LDA matrix ({}-dim)...'.format(self.LDAdim))
        print('\ttraining LDA matrix (%dD >> %dD)...' % (self.rawDim, self.LDAdim))
        global_mean = np.zeros((self.rawDim, 1), dtype='float32')
        class_means = dict()
        Sw = np.zeros((self.rawDim, self.rawDim), dtype='float32')
        Sb = np.zeros((self.rawDim, self.rawDim), dtype='float32')
        n_dvecs = 0
        for dvec_path in pathlist:
          dvec_dict = load_dict_npy(dvec_path)
          spkIDlist = list(dvec_dict.keys())

          ## Centering #@!
          for spkID in spkIDlist:
            dvec_dict[spkID] -= self.m0

          ## Accumulate a global mean embedding
          dvec_stack = np.hstack(dict2list(dvec_dict))
          global_mean += np.sum(dvec_stack, axis=1, keepdims=True)
          n_dvecs += dvec_stack.shape[1]

          ## Accumulate class mean embeddings
          for spkID in spkIDlist:
            class_means[spkID] = dvec_dict[spkID].mean(axis=1, keepdims=True)

          ## Accumulate intra-class covariance (Sw)
          for spkID in spkIDlist:
            cen_data = dvec_dict[spkID] - class_means[spkID]
            Sw += np.dot(cen_data, cen_data.T) / cen_data.shape[1]

        ## Compute global mean
        global_mean /= n_dvecs

        ## Compute inter-class covariance (Sb)
        for spkID in class_means:
          class_means[spkID] -= global_mean
          Sb += np.dot(class_means[spkID], class_means[spkID].T)

        ## Compute LDA matrix
        try:
          Dmat = np.linalg.solve(Sw, Sb)
        except:
          Dmat = np.linalg.lstsq(Sw, Sb)
        evals, evecs = np.linalg.eigh(Dmat)
        evals, evecs = evals.real, evecs.real

        inds = evals.argsort()[-self.LDAdim:][::-1]
        self.LDAmat = evecs[:, inds].T

        ## Reallocate parameters to match the reduced data dimension
        self.dataDim = self.LDAdim
        self.B = np.eye(self.dataDim, dtype='float32')
        self.W = np.eye(self.dataDim, dtype='float32')


      ### 3) WCCN ###
      if self.do_WCCN:
        print('\tcomputing WCCN matrix...')
        ## Compute intra-class covariance (Sw)
        n_spks = 0
        Sw = np.zeros((self.dataDim, self.dataDim), dtype='float32')
        for dvec_path in pathlist:
          dvec_dict = load_dict_npy(dvec_path)
          spkIDlist = list(dvec_dict.keys())

          ## Centering + LDA #@!
          for spkID in spkIDlist:
            dvec_dict[spkID] = np.dot(self.LDAmat, 
                                      dvec_dict[spkID] - self.m0)

          ## Accumulate for intra-class covariance (Sw)
          n_spks += len(spkIDlist)
          for spkID in spkIDlist:
            cen_data = dvec_dict[spkID] - dvec_dict[spkID].mean(axis=1, keepdims=True)
            Sw += np.dot(cen_data, cen_data.T) / cen_data.shape[1]
        Sw = Sw / n_spks

        ## Compute WCCN matrix
        self.B = np.linalg.cholesky(np.linalg.inv(Sw)).T


      ### 4) Whitening ###
      if self.do_whitening:
        print('\tcomputing whitening matrix...')
        ## Compute global mean
        n_dvecs = 0
        self.m1 = np.zeros((self.dataDim, 1), dtype='float32')
        for dvec_path in pathlist:
          dvec_dict = load_dict_npy(dvec_path)
          spkIDlist = list(dvec_dict.keys())

          ## Centering + LDA + WCCN #@!
          for spkID in spkIDlist:
            dvec_dict[spkID] = np.dot(self.B, 
                                      np.dot(self.LDAmat, 
                                             dvec_dict[spkID] - self.m0))

          ## Accumulate for global mean
          dvec_stack = np.hstack(dict2list(dvec_dict))
          self.m1 += np.sum(dvec_stack, axis=1, keepdims=True)
          n_dvecs += dvec_stack.shape[1]
        self.m1 /= n_dvecs

        ## Compute Cov_tot
        Cov_tot = np.zeros((self.dataDim, self.dataDim), dtype='float32')
        for dvec_path in pathlist:
          dvec_dict = load_dict_npy(dvec_path)
          spkIDlist = list(dvec_dict.keys())

          ## Centering + LDA + WCCN + Centering (to obtain "Cov_tot") #@!
          for spkID in spkIDlist:
            dvec_dict[spkID] = np.dot(self.B, 
                                      np.dot(self.LDAmat, 
                                             dvec_dict[spkID] - self.m0)) \
                             - self.m1
            Cov_tot += np.dot(dvec_dict[spkID], dvec_dict[spkID].T)
        Cov_tot = Cov_tot / (n_dvecs - 1)

        ## Compute whitening matrix
        evals, evecs = np.linalg.eigh(Cov_tot)
        evals, evecs = evals.real, evecs.real
        ## =====  1) PLDA_package  ===== ##
        self.W = np.dot(np.diag(1./np.sqrt(evals)), evecs.T)
        ## =====  2) SIDEKIT  ========== ##
        # ind = evals.argsort()[::-1]
        # evals = evals[ind]
        # evecs = evecs[:,ind]
        # self.W = np.dot(np.diag(1/np.sqrt(evals)), evecs.T)
        if np.isnan(self.W).sum(): raise ValueError('NaN exists in W!!')
        if np.isinf(self.W).sum(): raise ValueError('Inf exists in W!!')

    else:
      if isinstance(pathlist, np.ndarray): # ndarray in (ivDim, n_utts)
        ndim = pathlist.ndim
        if ndim == 1:
          pathlist = pathlist[:, None]
        dvecs = np.dot(self.W, 
                np.dot(self.B, 
                np.dot(self.LDAmat, pathlist - self.m0)) - self.m1) - self.mu
        dvecs = length_norm(dvecs) if self.do_lengthnorm else dvecs
        if ndim == 1:
          dvecs = dvecs.squeeze(axis=1)
        return dvecs
        # return length_norm(dvecs) if self.do_lengthnorm else dvecs

      elif isinstance(pathlist, dict):  # dict == {spkID1:ndarray1, ...}
        return {spkID:self.preprocess_steps(dvecs) \
                for spkID, dvecs in pathlist.items()}

      elif isinstance(pathlist, list):  # list == [ndarray1, ndarray2, ...]
        return [self.preprocess_steps(dvecs) for dvecs in pathlist]


  #### ========================================================================= ####

  def compute_suffstats(self, pathlist):
    N = 0
    S = np.zeros((self.dataDim, self.dataDim), dtype='float64')
    for dvec_path in sorted(pathlist):
      ## Centering + LDA + WCCN + Whitening #@!
      dvec_dict = self.preprocess_steps(load_dict_npy(dvec_path))

      ## Compute f (first-order)
      suff_path = dvec_path.replace('.'+self.ext, '.suff.'+self.ext)
      f_dict = {spkID:np.sum(dvecs, axis=1, keepdims=True) \
                for spkID, dvecs in dvec_dict.items()}
      save_dict_npy(suff_path, f_dict)

      ## Compute N and S (zeroth- and second-order)
      dvec_list = dict2list(dvec_dict)
      dvec_stack = np.hstack(dvec_list)
      N += dvec_stack.shape[1] # num_utts
      S += np.dot(dvec_stack, dvec_stack.T)
    return N, S


  def initialize(self, pathlist, N, S):
    # self.mu = np.zeros((self.dataDim, 1))
    ## ========================================= ##
    ## ===============    PLDA    ============== ##
    ## ========================================= ##
    if self.model_type in ['simp', 'std']:
      ## =========================== ##
      ## ==== Initialize self.V ==== ##
      ## =========================== ##
      ## 1) MSR style
      self.V = np.random.randn(self.dataDim, self.Vdim)
      self.V -= self.V.mean(axis=1)[:,None]
      U_svd, S_svd, V_svd = np.linalg.svd(self.V.T.dot(self.V))
      W_for_V = np.dot(V_svd.T, np.diag(1./(np.sqrt(S_svd)+1e-10)))
      self.V = np.dot(self.V, W_for_V)
      ## 2) SIDEKIT style
      # evals, evecs = np.linalg.eigh(S/N)
      # ind = evals.real.argsort()[::-1][:self.Vdim]
      # evals = evals.real[ind]
      # evecs = evecs.real[:,ind]
      # W_for_V = np.dot(evecs, np.diag(1/np.sqrt(evals.real)))
      # self.V = np.dot(self.V, W_for_V)

      ## =========================== ##
      ## ==== Initialize self.U ==== ##
      ## =========================== ##
      ## 1) MSR style
      # self.U -= self.U.mean(axis=1)[:,np.newaxis]
      # (U_svd, S_svd, V_svd) = np.linalg.svd(self.U.T.dot(self.U))
      # W_for_U = np.dot( V_svd.T, np.diag(1/(np.sqrt(S_svd)+1e-10)) )
      # self.U = self.U.dot(W_for_U)
      ## 2) Random initialization (PLDA_package)
      self.U = np.random.randn(self.dataDim, self.Udim)

      ## ============================= ##
      ## === Initialize self.Sigma === ##
      ## ============================= ##
      if self.SIGMA_INIT_METHOD == 'covariance':
        ## PLDA_package && SIDEKIT style
        self.Sigma = self.SIGMA_SCALE * S/N
      elif self.SIGMA_INIT_METHOD == 'random':
        ## PLDA_package style
        S_random = np.random.randn(self.dataDim, self.dataDim) / self.dataDim
        self.Sigma = np.dot(S_random, S_random.T)
      else:
        raise RuntimeError('Unknown init_Sigma_method')

      if self.model_type == 'std':  # standard PLDA -> diagonilize
        ## PLDA_package style
        self.Sigma = np.diag(np.diagonal(self.Sigma))

    ## ========================================= ##
    ## ===============    TCov    ============== ##
    ## ========================================= ##
    elif self.model_type in ['tcov']:
      n_dvecs = 0
      self.mu = np.zeros((self.dataDim, 1), dtype='float64')
      for dvec_path in sorted(pathlist):
        ## Centering + LDA + WCCN + Whitening #@!
        dvec_dict = self.preprocess_steps(load_dict_npy(dvec_path))
        dvec_stack = np.hstack(dict2list(dvec_dict))
        n_dvecs += dvec_stack.shape[1]
        self.mu += np.mean(dvec_stack, axis=1, keepdims=True)
      self.mu = self.mu / n_dvecs

      ## 1) JB-style
      # data_ms = [spk - self.mu for spk in data]
      # pooled_spk_mean = np.hstack([spk.mean(axis=1, keepdims=True) for spk in data_ms])
      # self.S_mu = np.linalg.inv(np.dot(pooled_spk_mean, pooled_spk_mean.T) / len(data_ms))

      # pooled_lms = np.hstack([spk - spk.mean(axis=1, keepdims=True) for spk in data_ms])
      # self.S_eps = np.linalg.inv(np.dot(pooled_lms, pooled_lms.T) / N

      ## 2) Package-style
      # sample_cov = S/N - np.dot(self.mu, self.mu.T)
      # self.S_mu = np.linalg.inv(sample_cov)
      # self.S_eps = np.linalg.inv(sample_cov)

      ## 3) Random initialization
      S_random = np.random.randn(self.dataDim, self.dataDim) / self.dataDim
      self.S_mu = np.dot(S_random, S_random.T)
      self.S_eps = np.dot(S_random, S_random.T)

    ## ========================================= ##
    ## ================    JB    =============== ##
    ## ========================================= ##
    elif self.model_type in ['jb']:
      ## 1) Sample-driven initialization
      n_spks = 0
      pooled_spk_mean = []
      self.S_eps = np.zeros((self.dataDim, self.dataDim), dtype='float64')
      for dvec_path in sorted(pathlist):
        dvec_dict = self.preprocess_steps(load_dict_npy(dvec_path))
        dvec_list = dict2list(dvec_dict)
        n_spks += len(dvec_list)
        pooled_spk_mean += [np.hstack(
          [dvec.mean(axis=1, keepdims=True) for dvec in dvec_list]
        )]
        pooled_lms = np.hstack(
          [dvec - dvec.mean(axis=1, keepdims=True) for dvec in dvec_list]
        )
        self.S_eps += np.dot(pooled_lms, pooled_lms.T)
      pooled_spk_mean = np.hstack(pooled_spk_mean)
      self.S_mu = np.dot(pooled_spk_mean, pooled_spk_mean.T) / n_spks
      self.S_eps = self.S_eps / N

      ## 2) Random initialization
      # S_random = np.random.randn(self.dataDim, self.dataDim) / self.dataDim
      # self.S_mu = np.dot(S_random, S_random.T)
      # self.S_eps = np.dot(S_random, S_random.T)


  def train(self, pathlist, N, S, n_iters=3):
    pathlist.sort()
    for it in np.arange(n_iters):
      ## Gaussian PLDAs
      if self.model_type in ['simp', 'std']:
        T_y, T_x, R_yy, R_yx, R_xx, Y_md = self.plda_e_step(pathlist, N, S)
        self.plda_m_step(R_yy, R_yx, R_xx, T_y, T_x, N, S)
        if self.do_md_step:
          self.plda_md_step(R_yy, R_yx, R_xx, Y_md, N)

      ## Two-Covariance model
      elif self.model_type in ['tcov']:
        T, R, Y = self.tcov_e_step(pathlist, N, S)
        self.tcov_m_step(T, R, Y, N, S)

      ## Joint Bayesian model
      elif self.model_type in ['jb']:
        self.jb_em_step(pathlist, N)

      ## Compute marginal log-likelihood (EM objective)
      llk_m_avg = self.compute_llk(pathlist)
      print('Iter {:02d}, llk = {:.7f}'.format(it+1, llk_m_avg))
    ## Compute params for scoring
    self.compute_Gamma_Lambda_tot()


  #### ========================================================================= ####




  #### ========================================================================= ####
  #### ==================   E-M Algorithms for Training LGMs  ================== ####
  #### ========================================================================= ####

  def plda_e_step(self, pathlist, N, S):
    V = self.V
    U = self.U
    Lambda = np.linalg.inv(self.Sigma)

    ## Set auxiliary matrices
    T = np.zeros((self.Vdim+self.Udim, self.dataDim))
    R_yy = np.zeros((self.Vdim, self.Vdim))
    Ey = np.zeros((self.Vdim, self.n_clas))
    Y_md = np.zeros((self.Vdim, self.Vdim))

    if self.Udim > 0:
      Q = np.linalg.inv(reduce(np.dot, [U.T, Lambda, U]) + np.eye(self.Udim))
      J = reduce(np.dot, [U.T, Lambda, V])
      H = V - reduce(np.dot, [U, Q, J])
    else:
      H = V

    LH = np.dot(Lambda, H)
    VLH = np.dot(V.T, LH)

    f_hstack = []
    i = 0
    n_previous = 0  # number of utterances for a previous person
    for dvec_path in pathlist:
      ## Centering + LDA + WCCN + Whitening #@!
      dvec_dict = self.preprocess_steps(load_dict_npy(dvec_path))
      ## Load first-order sufficient statistics
      suff_path = dvec_path.replace('.'+self.ext, '.suff.'+self.ext)
      f_dict = load_dict_npy(suff_path)
      ## Perform E-step
      for spkID, dvecs in sorted(dvec_dict.items(), 
                                 key=lambda kv: kv[1].shape[1]):
        n = dvecs.shape[1]
        if n != n_previous:
          ## Update matrices that are dependent on the number of utterances
          M_i = np.linalg.inv(n*VLH + np.eye(self.Vdim))
          n_previous = n

        f_i = f_dict[spkID]
        f_hstack.append(f_i)
        Ey[:, [i]] = np.dot(M_i, np.dot(LH.T, f_i))
        Eyy = np.dot(Ey[:, [i]], Ey[:, [i]].T)
        Y_md += M_i + Eyy   # it's for the MD-step
        R_yy += n*(M_i + Eyy)
        i += 1
    f_hstack = np.hstack(f_hstack)

    Y_md = Y_md / self.n_clas
    T_y = np.dot(Ey, f_hstack.T)  # T_y = Ey * f'

    if self.Udim > 0:
      ## T_x = Q * (U'*Lambda*S - J*T_y)
      T_x = np.dot(Q, reduce(np.dot, [U.T, Lambda, S]) \
          - np.dot(J, T_y))
      ## R_yx = (T_y*Lambda*U - R_yy*J')*Q 
      R_yx = np.dot(reduce(np.dot, [T_y, Lambda, U]) \
           - np.dot(R_yy, J.T), Q) 

      ## Auxiliary matrices
      W1 = np.dot(Lambda, U)
      W2 = np.dot(J, T_y)

      ## R_xx = Q*(W1'*S*W1 - W1'*W2' - W2*W1 + J*R_yy*J')*Q + N*Q;
      W3 = reduce(np.dot, [W1.T, S, W1]) - np.dot(W1.T, W2.T) \
         - np.dot(W2,W1) + reduce(np.dot, [J, R_yy, J.T])
      R_xx = reduce(np.dot, [Q, W3, Q]) + N*Q 
    else:
      T_x, R_yx, R_xx = (None, None, None)
    return (T_y, T_x, R_yy, R_yx, R_xx, Y_md)

  def plda_m_step(self, R_yy, R_yx, R_xx, T_y, T_x, N, S):
    if T_x is not None:
      T = np.vstack([T_y, T_x])
    else:
      T = T_y

    if self.Udim > 0:
      ## R == [R_yy, R_yx; R_yx', R_xx]
      R = np.vstack([np.hstack([R_yy, R_yx]), 
                     np.hstack([R_yx.T, R_xx])])
    else:
      R = R_yy
    
    VU = np.linalg.solve(R.T, T).T  # VU = T'/R;

    self.V = VU[:, :self.Vdim].copy()
    self.U = VU[:, self.Vdim:].copy()
    
    Sigma = (S - np.dot(VU, T)) / N
    
    ## Check for the PLDA type
    if self.model_type == 'std':
      self.Sigma = np.diag(np.diagonal(Sigma))
    else:
      self.Sigma = Sigma

  def plda_md_step(self, R_yy, R_yx, R_xx, Y_md, N):
    if self.Udim > 0:
      G = np.linalg.solve(R_yy.T, R_yx).T  # G = R_yx' / R_yy;
      X_md = (R_xx - np.dot(G, R_yx)) / N
      self.U = np.dot(self.U, np.linalg.cholesky(X_md))
      self.V = np.dot(self.V, np.linalg.cholesky(Y_md)) \
             + np.dot(self.U, G)
    else:
      self.V = np.dot(self.V, np.linalg.cholesky(Y_md))

  #### ========================================================================= ####

  def tcov_e_step(self, pathlist, N, S):
    Bcov = np.linalg.inv(self.S_mu)
    Wcov = np.linalg.inv(self.S_eps)
    mu = self.mu

    ## Initialize output matrices
    T = np.zeros((self.dataDim, self.dataDim))
    R = np.zeros((self.dataDim, self.dataDim))
    Y = np.zeros((self.dataDim, 1))

    ## Set auxiliary matrix
    Bmu = np.dot(Bcov, mu)

    n_previous = 0  # num_utts for the previous speaker
    for dvec_path in pathlist:
      ## Centering + LDA + WCCN + Whitening #@!
      dvec_dict = self.preprocess_steps(load_dict_npy(dvec_path))
      ## Load first-order sufficient statistics
      suff_path = dvec_path.replace('.'+self.ext, '.suff.'+self.ext)
      f_dict = load_dict_npy(suff_path)
      ## Perform E-step
      for spkID, dvecs in sorted(dvec_dict.items(), 
                                 key=lambda kv: kv[1].shape[1]):
        n = dvecs.shape[1]
        if n != n_previous:
          invL_i = np.linalg.inv(Bcov + n*Wcov)
          n_previous = n

        f_i = f_dict[spkID]
        gamma_i = Bmu + np.dot(Wcov, f_i)
        Ey_i = np.dot(invL_i, gamma_i)
        T += np.dot(Ey_i, f_i.T)
        R += n * (invL_i + np.dot(Ey_i, Ey_i.T))
        Y += n * Ey_i
    return (T, R, Y)

  def tcov_m_step(self, T, R, Y, N, S):
    self.mu = Y / N
    self.S_mu = (R - np.dot(Y, Y.T)/N) / N
    self.S_eps = (S - (T + T.T) + R) / N

  #### ========================================================================= ####

  def jb_em_step(self, pathlist, N):
    S_mu = self.S_mu
    S_eps = self.S_eps

    F = np.linalg.pinv(S_eps)
    S_mu_F = np.dot(S_mu, F)

    E_mu = np.zeros((self.dataDim, self.n_clas))
    E_eps = np.zeros((self.dataDim, self.dataDim))

    i = 0
    n_previous = 0  # num_utts for the previous speaker
    for dvec_path in pathlist:
      ## Centering + LDA + WCCN + Whitening #@!
      dvec_dict = self.preprocess_steps(load_dict_npy(dvec_path))
      ## Load first-order sufficient statistics
      suff_path = dvec_path.replace('.'+self.ext, '.suff.'+self.ext)
      f_dict = load_dict_npy(suff_path)
      ## Perform E-step
      for spkID, dvecs in sorted(dvec_dict.items(), 
                                 key=lambda kv: kv[1].shape[1]):
        n = dvecs.shape[1]
        if n != n_previous:
          G = -np.linalg.pinv(n*S_mu + S_eps).dot(S_mu_F)
          S_mu_F_mG = np.dot(S_mu, F + n*G)
          S_eps_G = np.dot(S_eps, G)
          n_previous = n

        f_i = f_dict[spkID]
        E_mu[:, [i]] = np.dot(S_mu_F_mG, f_i)

        S_eps_G_f = np.dot(S_eps_G, f_i)  # vector
        E_eps_i = dvecs + S_eps_G_f
        E_eps += np.dot(E_eps_i, E_eps_i.T)
        i += 1

    ## Perform M-step
    self.S_mu = np.dot(E_mu, E_mu.T) / self.n_clas
    self.S_eps = E_eps / N

