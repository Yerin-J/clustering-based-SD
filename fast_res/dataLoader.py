import torch, numpy, random, os, math, glob, soundfile
from torch.utils.data import Dataset, DataLoader
from scipy import signal

def loadWAV(filename, max_frames):
    max_audio = max_frames * 160 + 240 # 240 is for padding, for 15ms since window is 25ms and step is 10ms.
    audio, _ = soundfile.read(filename)
    audiosize = audio.shape[0]
    if audiosize <= max_audio: # Padding if the length is not enough
        shortage    = math.floor( ( max_audio - audiosize + 1 ) / 2 )
        audio       = numpy.pad(audio, (shortage, shortage), 'wrap')
        audiosize   = audio.shape[0]
    startframe = numpy.int64(random.random()*(audiosize-max_audio)) # Randomly select a start frame to extract audio
    feat = numpy.stack([audio[int(startframe):int(startframe)+max_audio]],axis=0)
    return feat

def loadWAV_speechaug(filename, max_frames):
    max_audio = max_frames * 160 + 240 # 240 is for padding, for 15ms since window is 25ms and step is 10ms.
    aug_audio = (max_frames//random.randint(5, 100))* 160 + 240
    snr_fix_ratio = aug_audio/max_audio
    audio, _ = soundfile.read(filename)
    audiosize = audio.shape[0]
    if audiosize <= max_audio: # Padding if the length is not enough
        shortage    = math.floor( ( max_audio - audiosize + 1 ) / 2 )
        audio       = numpy.pad(audio, (shortage, shortage), 'wrap')
        audiosize   = audio.shape[0]

    startframe = numpy.int64(random.random()*(audiosize-aug_audio)) # Randomly select a start frame to extract audio
    segment_audio = audio[int(startframe):int(startframe)+aug_audio]
    audiosize = segment_audio.shape[0]
    shortage = math.floor( ( max_audio - audiosize + 1 ) / 2 )
    p = random.random()
    if p < 0.2:
        segment_audio = numpy.pad(segment_audio, (0, 2*shortage), mode='constant', constant_values=0)
    else:
        segment_audio = numpy.pad(segment_audio, (2*shortage, 0), mode='constant', constant_values=0)
    feat = numpy.stack([segment_audio],axis=0)
    return feat, snr_fix_ratio

class train_loader(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, max_frames, **kwargs):
        self.train_path = train_path
        self.max_frames = max_frames
        self.max_audio = max_frames * 160 + 240 # Length of segment for training
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
        self.noiselist = {}
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        # self.rir_files = numpy.load('rir.npy') # Load the rir file
        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'));

        # Read training files
        with open(train_list) as dataset_file:
            lines = dataset_file.readlines();
        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split('-')[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        # Parse the training list into file names and ID indices
        self.data_list  = []
        self.data_label = []
        
        for lidx, line in enumerate(lines):
            data = line.strip().split();

            speaker_label = dictkeys[data[0].split('-')[0]];
            filename = os.path.join(train_path,data[1]);
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, index):
        segment = loadWAV(self.data_list[index], self.max_frames)
        p = random.random()
        if p < 0.5:
            segment = self.add_speech(segment, 'speech', frames = self.max_frames) # speech Noise

        q = random.random()
        if q < 0.2:
            segment = self.add_rev(segment, length = self.max_audio) # Rever
        elif q < 0.4:
            segment = self.add_noise(segment, 'music', frames = self.max_frames) # music Noise
        elif q < 0.6:
            segment = self.add_noise(segment, 'speech', frames = self.max_frames) # speech Noise
        elif q < 0.8:
            segment = self.add_noise(segment, 'noise', frames = self.max_frames) # noise Noise

        return torch.FloatTensor(segment), self.data_label[index]


    def __len__(self):
        return len(self.data_list)

    def add_rev(self, audio, length): 
        rir_file    = random.choice(self.rir_files)
        
        # rir_gains = numpy.random.uniform(-7,3,1)
        # rir     = numpy.multiply(rir_file, pow(10, 0.1 * rir_gains))
        rir, fs     = soundfile.read(rir_file)

        rir         = numpy.expand_dims(rir.astype(numpy.float),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:length]


    def add_speech(self, audio, noisecat, frames):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
        numnoise    = self.numnoise[noisecat]
        noisefile   = random.sample(self.noiselist[noisecat], 1)
        noiseaudio, snr_fix_ratio  = loadWAV_speechaug(noisefile[0], frames)
        noise_snr   = random.uniform(0, 10)
        noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)/snr_fix_ratio+1e-4) 
        noise = numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio
        p = random.random()
        if p < 0.8:        
            noise = self.add_rev(noise, length = self.max_audio)
        return noise + audio


    def add_noise(self, audio, noisecat, frames):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio  = loadWAV(noise, frames)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
        return numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True) + audio

# def worker_init_fn(worker_id):
#     numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

def get_loader(args): # Define the data loader
    trainLoader = train_loader(**vars(args))
    # sampler = torch.utils.data.DistributedSampler(trainLoader, shuffle=True)
    trainLoader = torch.utils.data.DataLoader(
        trainLoader,
        # sampler=sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        drop_last=True,
        # worker_init_fn=worker_init_fn,
        prefetch_factor=5,
    )
    return trainLoader
