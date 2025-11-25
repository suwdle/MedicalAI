import random

def generate_triplets(sequences: List[str], num_pairs: int):
    rows = []
    print(f"Generating {num_pairs} triplets")
    for _ in range(num_pairs):
        anchor = random.choice(sequences)
        if len(anchor) > MAX_LENGTH:
            start = random.randint(0, len(anchor) - MAX_LENGTH)
            anchor = anchor[start:start+MAX_LENGTH]
        
        if random.random() < 0.5:
            pos = anchor
        else:
            pos = reverse_complement(anchor)
            
        n_mut = random.randint(3, 15)
        neg = apply_biological_snv(anchor, n_mut)
        
        rows.append([anchor, pos, neg])
    return rows