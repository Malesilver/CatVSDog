import time
import torch as t
from multiprocessing import cpu_count

# [data path]
TRAIN_IMG_PATH = "./data/train"
TEST_IMG_PATH = "./data/test"

# [image info]
RE_HEIGHT = 200
RE_WIDTH = 200

# [train setting]
EX_NAME = "ex01-Alexnet"
MODEL = "Simplenet"

GPU = "1"
BATCH_SIZE = 400
LR = 1e-1
MAX_EPOCH = 10000
WORKERS = 0 # cpu_count()

DEVICE = t.device("cuda:" + GPU if t.cuda.is_available() else "cpu")
TIME = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
SAVE_PATH = "/home/lb/CatVSDog/Result/{}/{}__GPU_{}-LR_{}-[{}@{}@{}]-MODEL-{}".format(EX_NAME, TIME,GPU, LR, BATCH_SIZE, RE_HEIGHT, RE_WIDTH, MODEL)

# [test setting]
WEIGHTS = "./Result/ex01-Alexnet/2018-11-12_13-33-28__GPU_1-LR_0.1-[400@200@200]-MODEL-Simplenet/model/000best_val_loss_0.1438_model.pt"

# [result setting]
TEST_SAVE_PATH = "./submission.csv"