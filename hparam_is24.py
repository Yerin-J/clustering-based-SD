
class Hyperparams:

  ## ---------------------------------------------------------- ##
  """ Acoustic feature extraction """
  stft_opts_torch = dict(
    n_fft=512, hop_length=160, win_length=400, win_type='hanning', 
    symmetric=True)

  mel_opts_torch = dict(
    fs=16000, nfft=512, lowfreq=64., maxfreq=8000., 
    nlinfilt=0, nlogfilt=64)

  ## ---------------------------------------------------------- ##
  """ DataLoader """
  seed = 0
  num_parallel_jobs = 8
  max_queue_size = 10

  minibatch_opts = dict(
    spb=32, # number of speakers per batch
    ups=1,  # number of utterances per speaker
    ch_incre=1,  # channel increment
    croppy=True, # whether to randomly crop utterances
    minLen=250,  # lower bound for cropped segment length
    maxLen=250,  # upper bound for cropped segment length
    start_random=True, # whether to start crop from the beginning
  )

  ## Stacking opts: [(key, func, kwargs), ...]
  import numpy as np
  # stack_opts = [('mfbe40', np.stack, {'axis':0})]
  stack_opts = [('mfbe', np.stack, {'axis':0}), 
                ('spkidx', np.hstack, {})]

  ## ---------------------------------------------------------- ##
  """ Model """
  feature_dim = 64  # dimension of the input feature
  feature_normalizer = "swcmn"  # None, "swcmn", "gmvn"
  if feature_normalizer == "swcmn":
    sliding_window_size = 300


  ## BLSTM-based feature augmentation
  feat_aug_opts = dict(
    num_aug=0)

  ## ResBlock configs
  blocks_per_level = [3, 4, 6, 3]

  kernel_opts = {\
    "conv0": (48, [3, 3], True),
    "resblock1": (48, [3, 3], True),
    "resblock2": (96, [3, 3], True),
    "resblock3": (192, [3, 3], True),
    "resblock4": (384, [3, 3], True),
    "shortcut_bn": True,
  }
  sqzext, reduction_factor = (True, 4)

  ## Bias / Weight init config
  use_bias = False
  kernel_init_scale = 1.0  # gain for variance-scaling initalizer

  ## Batch-norm config
  batchnorm_momentum = 0.99  # batch-norm momentum for moving average
  batchnorm_epsilon = 1e-6   # batch-norm epsilon for variance

  ## Activation function
  activ_func, activ_kwargs = "ReLU", dict()
  # activ_func, activ_kwargs = "LeakyReLU", dict(negative_slope=0.2)


  ## Pooling options
  l2_normalize_stats = True
  use_global_info = True
  allow_shortcuts = [True, True, True, True, True]
  attention_layer_dims = [16, 16, 32, 64, 128]

  ## Fully-connected layer for speaker embedding
  fc_layer_units = 256
  fc_layer_batchnorm = True
  fc_layer_bias = True
  fc_layer_trainable = True

  ## Embedding node
  embedding_node = 'fc_layer_bn'  # Name of the embedding node
  # embedding_node = 'fc_layer_affine'  # Name of the embedding node


  ## ---------------------------------------------------------- ##
  """ Loss layer """
  loss_layer_trainable = True

  ## Softmax loss type
  # loss_type = "softmax"
  loss_type = "amsoftmax"
  # loss_type = "aamsoftmax"

  ## Linearity of activ func of the penultimate layer
  if loss_type in ["softmax"]:
    penulti_linear = False
  else:
    penulti_linear = True

  loss_opts, anneal_opts = dict(normalize=False), {}
  if loss_type == 'asoftmax':
    loss_opts = dict(s=20.0, m=4, eps=1e-12)
    loss_opts = dict(s=0.0, m=4, eps=1e-12)
    anneal_opts = dict(
      # lmbd_min=10., lmbd_base=1000., lmbd_gamma=0.0005, lmbd_power=5)
      lmbd_min=0., lmbd_base=1000., lmbd_gamma=0.00001, lmbd_power=5)
  elif loss_type == 'amsoftmax':
    loss_opts = dict(s=20.0, m=0.2, eps=1e-12)
    loss_opts = dict(s=0.0, m=0.2, eps=1e-12)
    anneal_opts = dict(
      # lmbd_min=0., lmbd_base=1000., lmbd_gamma=0.001, lmbd_power=5)
      lmbd_min=0., lmbd_base=1000., lmbd_gamma=0.0001, lmbd_power=5)
  elif loss_type == 'aamsoftmax':
    loss_opts = dict(s=30.0, m=0.4, eps=1e-12)
    loss_opts = dict(s=0.0, m=0.4, eps=1e-12)
    anneal_opts = dict(
      # lmbd_min=0., lmbd_base=1000., lmbd_gamma=0.0001, lmbd_power=5)
      lmbd_min=0., lmbd_base=1000., lmbd_gamma=0.00001, lmbd_power=5)

  ## Auxiliary loss function
  aux_loss_type = None # "mhe_loss", "bsb_loss"
  aux_loss_scale = 0.01


  ## ---------------------------------------------------------- ##
  """ Training configs """
  ## For large dataset
  num_train_epochs = 100  # number of training epochs
  num_iters_per_epoch = 30000
  num_iters_per_summary = 300
  ## For small dataset
  # num_train_epochs = 100  # number of training epochs
  # num_iters_per_epoch = 2000
  # num_iters_per_summary = 100

  ## Early stopping epochs
  num_early_stop_epochs = 8

  ## Optimizer
  optimizer = "sgd"  # "sgd", "momentum", "adam"
  # optimizer = "momentum"  # "sgd", "momentum", "adam"
  # optimizer = "adam"  # "sgd", "momentum", "adam"

  ## Learning rate & decay
  if optimizer == "sgd":
    learning_rate, learning_rate_decay = (0.01, 1/2.)
    optimizer_opts = dict()
  elif optimizer == "momentum":
    learning_rate, learning_rate_decay = (0.001, 1/2.)
    optimizer_opts = dict(momentum=0.9, nesterov=False)
  elif optimizer == "adam":
    learning_rate, learning_rate_decay = (0.0001, 1/2.)
    optimizer_opts = dict(betas=(0.9, 0.999), eps=1e-06, amsgrad=True)
  learning_rate_min = 1e-6

  ## Learning rate annealing
  anneal_lr_from_valid = True
  if anneal_lr_from_valid:
    num_valid_epochs_tolerate = 3
    num_offset_after_anneal = min(1, num_valid_epochs_tolerate)
  else:
    anneal_schedule = [30, 40, 48, 70] # [30, 40, 48, 70], [20, 30, 40, 50]

  ## L2-regularization of weights
  kernel_regularizer_l2 = 1e-2
  # projection_regularizer = 1e-3

  ## Gradient clipping
  clip_gradient = False
  clip_gradient_norm = 3.0

  ## Dropout config
  dropout_rate = 0.2     # dropout rate
  dropout_frequency = 3  # apply dropout for every N-th batch

  ## Resume training
  use_saved_lr = True
  learning_rate_forced = None


  ## ---------------------------------------------------------- ##
  """ PldaVariants.py """
  do_LinearRegr = False
  use_plda = True

  ### ============= ###
  ###  PLDA params  ###
  ### ============= ###
  plda_opts = dict(model_type='tcov', 
                   ## LDA projection opts ##
                   dataDim=fc_layer_units, LDAdim=192, 
                   ## PLDA projection opts ##
                   Vdim=0, Udim=0, 
                   ## Pre-processing opts ##
                   do_centering=True, 
                   do_WCCN=False, 
                   do_whitening=False, 
                   do_lengthnorm=True, 
                   do_mindiv=True)
  n_iters = 3
  overwrite_plda = False


  # ## Cosine similarity scoring
  # use_plda = False
  # plda_opts['LDAdim'] = 0
  # plda_opts['do_centering'] = False
  # plda_opts['do_centering'] = True
  # plda_opts['do_whitening'] = False


  need_preproc = any([use_plda, do_LinearRegr, 
                      any([plda_opts['do_centering'], 
                           plda_opts['LDAdim'], 
                           plda_opts['do_WCCN'], 
                           plda_opts['do_whitening']])])

  scoring_method = '[Emb(%d)-' % plda_opts['dataDim']
  if plda_opts['do_centering']:
    scoring_method += 'Center-'
  if plda_opts['LDAdim']:
    scoring_method += 'LDA(%d)-' % plda_opts['LDAdim']
  if plda_opts['do_WCCN']:
    scoring_method += 'WCCN-'
  if plda_opts['do_whitening']:
    scoring_method += 'Whiten-'
  if use_plda:
    if plda_opts['do_lengthnorm']:
      scoring_method += 'LN-'
    scoring_method += plda_opts['model_type'] + ']'
  else:
    scoring_method += 'CSS]'

