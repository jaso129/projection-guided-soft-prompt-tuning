# Config/config.py
import torch  # 新增此行

MODEL_NAME = "t5-base"
N_TOKENS = 10
PREFIX_LEN = 10
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_PATH = "Checkpoints/soft_prompts.pt"
SAMPLE_SIZE = -1  # 隨機取樣的樣本數量 (-1 表示不取樣)