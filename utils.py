import random
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_pretrained_model(device):
    MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(device).eval()
    
    return tokenizer, model