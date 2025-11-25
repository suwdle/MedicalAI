import os, random
import numpy as np
import pandas as pd
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

# ======================
# 하이퍼파라미터 / 설정
# ======================

SEED = 42
MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"

# 경로
DATA_DIR = "./data"
TEST_PATH = os.path.join("/content/drive/MyDrive/Dacon/Medical AI/test.csv")
SAMPLE_SUB_PATH = os.path.join("/content/drive/MyDrive/Dacon/Medical AI/sample_submission.csv")
OUT_PATH = "submission_v2.csv"

# 임베딩 관련
OUTPUT_DIM = 2048
LAST_N_LAYERS = 24   # 실제 레이어 수보다 많으면 자동으로 줄여서 사용
MAX_LENGTH = 512

# 학습/추론 설정
USE_FP16 = True
BATCH_SIZE_TR = 32
BATCH_SIZE_INFER = 32
NUM_WORKERS = 2

TRAIN_EPOCHS = 5
LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4
TRIPLET_MARGIN = 1.0
NUM_TRIPLETS_PER_EPOCH = 30000  # epoch마다 새로 생성


# ===============
# 유틸 함수들
# ===============

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)


_rc_map = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    return seq.translate(_rc_map)[::-1]


def apply_biological_snv(seq: str, num_mutations: int) -> str:
    """임의 위치의 염기를 SNV 방식으로 바꿔줌"""
    bases = ['A', 'C', 'G', 'T']
    s = list(seq)
    if len(s) == 0:
        return seq
    idx_list = random.sample(range(len(s)), min(num_mutations, len(s)))
    for idx in idx_list:
        orig = s[idx]
        candidates = [b for b in bases if b != orig]
        s[idx] = random.choice(candidates)
    return "".join(s)


def generate_triplets(sequences: List[str], num_pairs: int):
    """
    anchor, positive, negative triplet 생성
    - positive: anchor 그대로 또는 reverse complement
    - negative:
      - 80%: anchor에 1~5개 SNV만 가한 '작은' 변이
      - 20%: 완전히 다른 시퀀스 (다른 영역)
    """
    rows = []
    print(f"Generating {num_pairs} triplets")
    for _ in range(num_pairs):
        anchor = random.choice(sequences)
        if len(anchor) > MAX_LENGTH:
            start = random.randint(0, len(anchor) - MAX_LENGTH)
            anchor = anchor[start:start + MAX_LENGTH]

        # positive: anchor 자체 혹은 reverse complement
        pos = anchor if random.random() < 0.5 else reverse_complement(anchor)

        # negative 샘플링 전략
        if random.random() < 0.8:
            # 대부분은 작은 SNV 변이 (1~5개)
            r = random.random()
            if r < 0.7:
                n_mut = 1
            else:
                n_mut = random.randint(2, 5)
            neg = apply_biological_snv(anchor, n_mut)
        else:
            # 일부는 완전히 다른 서열
            neg = random.choice(sequences)
            if len(neg) > MAX_LENGTH:
                start = random.randint(0, len(neg) - MAX_LENGTH)
                neg = neg[start:start + MAX_LENGTH]

        rows.append([anchor, pos, neg])
    return rows


# =======================
# 모델 정의
# =======================

class RobustModel(nn.Module):
    def __init__(self, hidden_size: int, last_n: int, out_dim: int):
        super().__init__()
        self.last_n = last_n
        # 마지막 last_n개의 레이어를 가중합하기 위한 파라미터
        self.layer_weights = nn.Parameter(torch.zeros(last_n))

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, out_dim),
        )

    def forward(self, hidden_states: List[torch.Tensor], mask: torch.Tensor):
        """
        hidden_states: list of (B, L, H) / 전체 레이어
        mask: (B, L)  ; 1=유효, 0=pad
        """
        n_layers = len(hidden_states)
        use_n = min(self.last_n, n_layers)  # 실제 레이어 수보다 크지 않게 방어

        # 마지막 use_n개 레이어만 사용
        stack = torch.stack(hidden_states[-use_n:], dim=0)  # (use_n, B, L, H)
        w = F.softmax(self.layer_weights[:use_n], dim=0).view(-1, 1, 1, 1)
        feat = (stack * w).sum(dim=0)  # (B, L, H)

        # padding 제외 평균 pooling
        mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
        sum_embeddings = torch.sum(feat * mask_expanded, dim=1)  # (B, H)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # (B, 1)
        mean_emb = sum_embeddings / sum_mask  # (B, H)

        out = self.proj(mean_emb)  # (B, out_dim)
        return l2_normalize(out)


# =======================
# 메인 루틴
# =======================

def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --------- 데이터 로드 ---------
    test_df = pd.read_csv(TEST_PATH)
    sequences = test_df["seq"].tolist()
    print(f"Loaded {len(sequences)} sequences from test set")

    # --------- 토크나이저 / 백본 모델 ---------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    backbone.to(device)
    backbone.eval()

    # 백본은 freeze
    for p in backbone.parameters():
        p.requires_grad = False

    hidden_size = backbone.config.hidden_size
    model = RobustModel(hidden_size, LAST_N_LAYERS, OUTPUT_DIM).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)

    print("========== 학습 시작 ==========")
    model.train()

    bs = BATCH_SIZE_TR

    for epoch in range(TRAIN_EPOCHS):
        print(f"\n[Epoch {epoch + 1}/{TRAIN_EPOCHS}]")
        # ★ epoch마다 새로운 triplet 생성 → 더 다양한 변이 패턴 학습
        triplet_data = generate_triplets(sequences, NUM_TRIPLETS_PER_EPOCH)
        random.shuffle(triplet_data)

        epoch_loss = []

        for i in tqdm(range(0, len(triplet_data), bs)):
            batch = triplet_data[i:i + bs]
            if len(batch) < 2:
                continue

            anchors, poss, negs = zip(*batch)
            all_seqs = list(anchors) + list(poss) + list(negs)

            enc = tokenizer(
                all_seqs,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(device)

            with torch.cuda.amp.autocast(enabled=USE_FP16):
                with torch.no_grad():
                    out = backbone(**enc, output_hidden_states=True)

                embs = model(out.hidden_states, enc["attention_mask"])
                B = len(batch)
                ea, ep, en = embs[:B], embs[B:2 * B], embs[2 * B:]

                dist_pos = 1 - F.cosine_similarity(ea, ep)  # 작을수록 좋음
                dist_neg = 1 - F.cosine_similarity(ea, en)  # 클수록 좋음
                loss = F.relu(dist_pos - dist_neg + TRIPLET_MARGIN).mean()

            scaler.scale(loss).backward()

            # 선택: gradient clipping (폭주 방지용)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            epoch_loss.append(loss.item())

        print(f"Epoch {epoch + 1} mean loss: {np.mean(epoch_loss):.4f}")

    print("========== 추론 시작 ==========")
    model.eval()

    sub_df = pd.read_csv(SAMPLE_SUB_PATH)
    embeddings = []

    for i in tqdm(range(0, len(sequences), BATCH_SIZE_INFER)):
        batch = sequences[i:i + BATCH_SIZE_INFER]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = backbone(**enc, output_hidden_states=True)
            emb = model(out.hidden_states, enc["attention_mask"])
            embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    col_names = [f"emb_{i:04d}" for i in range(OUTPUT_DIM)]
    emb_df = pd.DataFrame(embeddings, columns=col_names)

    final_df = pd.concat([sub_df[["ID"]], emb_df], axis=1)
    final_df.to_csv(OUT_PATH, index=False)
    print(f"저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
