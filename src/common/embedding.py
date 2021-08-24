import csv
import os

from embeddings import GloveEmbedding, KazumaCharEmbedding
from tqdm import tqdm

from common.vocab import build_vocab


def build_embeddings(data_dir='data', enforce_refresh=False):
    path = os.path.join(data_dir, 'cache', 'embeddings.tsv')
    if not enforce_refresh and os.path.exists(path):
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            embeddings = [list(map(float, row)) for row in reader]
        return embeddings

    vocab = build_vocab()

    glove = GloveEmbedding(
        name='common_crawl_840',
        d_emb=300,
        show_progress=True,
    )
    kazuma = KazumaCharEmbedding(
        show_progress=True,
    )
    embeddings = []
    for token in tqdm(vocab.token2idx):
        emb1 = glove.emb(token, default='zero')
        emb2 = kazuma.emb(token, default='zero')
        e = emb1 + emb2
        embeddings.append(e)

    dump_dir = os.path.join(data_dir, 'cache')
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
    dump_path = os.path.join(dump_dir, 'embeddings.tsv')
    with open(dump_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(embeddings)

    return embeddings


if __name__ == '__main__':
    build_embeddings(
        enforce_refresh=True,
    )
