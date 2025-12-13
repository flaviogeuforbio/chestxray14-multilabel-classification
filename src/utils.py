from dataset import ChestXRayDataset, CLASSES
from torch import tensor

def get_pos_weights(dataset: ChestXRayDataset):
    df = dataset.df

    weights = []
    for c in CLASSES:
        pos_df = df[df['Finding Labels'].str.contains(c)]
        pos_samples = len(pos_df)
        neg_samples = len(df) - pos_samples
        weights.append(neg_samples / pos_samples)

    return tensor(weights)
