from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.ids  = df["ID"].tolist()
        self.seqs = df["seq"].tolist()
        self.tok  = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {"ID": self.ids[idx], "seq": self.seqs[idx]}

def collate_fn(batch, tok, max_len):
    ids  = [b["ID"] for b in batch]
    seqs = [b["seq"] for b in batch]
    enc  = tok.batch_encode_plus(
        seqs,
        return_tensors="pt",
        padding="longest",          
        truncation=True,
        max_length=max_len
    )
    # attention_mask: pad 토큰이 0
    return {
        "ids": ids,
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"]
    }