import sys, time, os, argparse, warnings, glob, torch
from tools import *
from model import *
from dataLoader import *
import dino_utils as utils

# Training settings
parser = argparse.ArgumentParser(description = "Stage I, Supervsied speaker recognition with AAM softmax.")
parser.add_argument('--batch_size',        type=int,   default=256,          help='Batch size, bigger is better')
parser.add_argument('--n_cpu',             type=int,   default=8,            help='Number of loader threads')
parser.add_argument('--optimizer',         type=str,   default="adam",       help='sgd or adam');
parser.add_argument('--test_interval',     type=int,   default=1,            help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',         type=int,   default=80,           help='Maximum number of epochs')
parser.add_argument('--lr',                type=float, default=0.001,        help='Learning rate')
parser.add_argument('--lr_min',            type=float, default=1e-6,         help='Minimum learning rate');
parser.add_argument("--warmup_epochs",     type=int,   default=10,           help="Number of epochs for the linear learning-rate warm up.")
parser.add_argument('--weight_decay',      type=float, default=1e-6,         help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
parser.add_argument('--weight_decay_end',  type=float, default=1e-6,         help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""")
parser.add_argument('--clip_grad',         type=float, default=3.0,          help="""Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""");
parser.add_argument('--initial_model',     type=str,   default="",           help='Initial model path')
parser.add_argument('--load_type',         type=str,   default=None,   choices=["init", "download", "load", None], help='Initial load_type')

################################################################################################################################################# data 경로 설정 필수!
parser.add_argument('--save_path',         type=str,   default="exp/exp",   help='Path for model and scores.txt')
parser.add_argument('--train_list',        type=str,   default="<<학습 시키고자 하는 data의 wav.scp 파일 경로>>>",           help='Path for dev list')
parser.add_argument('--val_list',          type=str,   default="<<trial set 파일인 veri_test.txt 경로 >>",     help='Evaluation list');
parser.add_argument('--train_path',        type=str,   default="<<학습 data (voxceleb) 경로>>",          help='Absolute path to the train set');
parser.add_argument('--val_path',          type=str,   default="<<validation set data (voxceleb) 경로>>", help='Absolute path to the test set');
#################################################################################################################################################

parser.add_argument('--max_frames',        type=int,   default=200)

## Model definition
parser.add_argument('--n_mels',            type=int,   default=80,          help='Number of mel filterbanks');
parser.add_argument('--nOut',              type=int,   default=256,         help='Embedding size in the last FC layer, resnet:256, ecapa-tdnn:192');

## Setting Classification
parser.add_argument('--n_class',           type=int,   default=1211,        help="""Dimensionality of the classification head output. Voxceleb1 : 1211, VoxCeleb2 5994.""")
parser.add_argument('--margin',            type=float, default=0.2,        help="""AAM softmax (ArcFace) margin.""")
parser.add_argument('--scale',             type=int,   default=30,          help="""AAM softmax (ArcFace) scale.""")

# Misc
parser.add_argument('--seed',              type=int,   default=0,           help='Random seed.')
parser.add_argument("--dist_url",          type=str,   default="env://",    help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--local_rank",        type=int,   default=0,           help="Please ignore and do not set this argument.")

parser.add_argument('--eval',              dest='eval', action='store_true', help='Do evaluation only')
args = parser.parse_args()

# Initialization
model_save_path     = args.save_path+"/model"
result_save_path    = args.save_path+"/result"
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")
os.makedirs(model_save_path, exist_ok = True)
scorefile = open(args.save_path+"/scores.txt", "a+")
it = 1

# utils.init_distributed_mode(args)
args.gpu = 0
torch.cuda.set_device(args.gpu)

utils.fix_random_seeds(args.seed)

trainLoader = get_loader(args) # Define the dataloader

Trainer = model(len(trainLoader)*args.batch_size+1, **vars(args)) # Define the framework
modelfiles = glob.glob('%s/model0*.model'%model_save_path) # Search the existed model files
modelfiles.sort()
        
if args.eval == True: # Do evaluation only
    if (args.initial_model != ""): # Otherwise, system will try to start from the saved model&epoch
        if args.load_type == "load":
            Trainer.load_network(args.initial_model)
        elif args.load_type == "download":
            Trainer.init_download_network(args.initial_model)
        elif args.load_type == "init":
            Trainer.init_network(args.initial_model)
        else:
            raise ValueError('Undefined pre-trained model load type')    
    elif len(modelfiles) >= 1:# If initial_model is exist, system will train from the initial_model
        Trainer.load_network(modelfiles[-1])
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
    
    EER, minDCF = Trainer.evaluate_network(**vars(args))
    print('EER %2.4f, minDCF %.3f\n'%(EER, minDCF))
    quit()
    
if len(modelfiles) >= 1:# If initial_model is exist, system will train from the initial_model
    Trainer.load_network(modelfiles[-1])
    it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
elif (args.initial_model != ""): # Otherwise, system will try to start from the saved model&epoch
    if args.load_type == "load":
        Trainer.load_network(args.initial_model)
    elif args.load_type == "download":
        Trainer.init_download_network(args.initial_model)
    elif args.load_type == "init":
        Trainer.init_network(args.initial_model)
    else:
        raise ValueError('Undefined pre-trained model load type')


while it <= args.max_epoch:
    # Train for one epoch
    loss, acc, lr = Trainer.train_network(loader=trainLoader, epoch = it)

    # Evaluation every [test_interval] epochs, record the training loss, training acc, evaluation EER/minDCF
    if it % args.test_interval == 0:
        Trainer.save_network(model_save_path+"/model%09d.model"%it)
        EER, minDCF = Trainer.evaluate_network(**vars(args))
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, LOSS %f, EER %2.4f, minDCF %.3f"%( lr, loss, EER, minDCF))
        scorefile.write("Epoch %d, LR %f, LOSS %f, EER %2.4f, minDCF %.3f\n"%(it, lr, loss, EER, minDCF))
        scorefile.flush()
    # Otherwise, recored the training loss and e:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, LOSS %f"%( lr, loss))
        scorefile.write("Epoch %d, LR %f, LOSS %f\n"%(it, lr, loss))
        scorefile.flush()

    it += 1
    print("")
quit()
