from datasets import load_dataset
import random
import random
import hashlib
from collections import defaultdict
from collections import deque
import numpy as np
from visualize import visualize_clusters

def get_ngrams(text, n=5):
    tokens = text.split()
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def hash_fn(seed):
    def h(x):
        return int(hashlib.md5((str(seed) + str(x)).encode()).hexdigest(), 16)
    return h                                                                                                     

def generate_signature(ngrams, num_hashes=128):
    hash_functions = [hash_fn(i) for i in range(num_hashes)]
    return [min(h(gram) for gram in ngrams) for h in hash_functions]



def lsh(signature, num_bands=16, rows_per_band=8):
    assert len(signature) == num_bands * rows_per_band
    buckets = []
    for b in range(num_bands):
        band = tuple(signature[b * rows_per_band : (b + 1) * rows_per_band])
        buckets.append(hash(band))
    return buckets



def generate_test_documents(total_unique_documents=700, total_num_repeated_documents=300, seed=42):
    """
    Generates a synthetic test dataset from real documents with duplicates.

    Args:
        total_unique_documents (int): Number of unique source documents to include
        total_num_repeated_documents (int): Number of repeated documents to inject
        seed (int): Random seed for reproducibility

    Returns:
        List[str]: All documents (unique + duplicates), shuffled
    """
    random.seed(seed)
    
    # Step 1: Load AG News dataset
    dataset = load_dataset("ag_news", split="train")
    
    # Step 2: Select unique documents
    docs = [item['text'] for item in dataset.select(range(total_unique_documents))]

    # Step 3: Add randomly repeated documents
    duplicate_docs = random.choices(docs, k=total_num_repeated_documents)
    
    # Step 4: Combine and shuffle
    all_docs = docs + duplicate_docs
    random.shuffle(all_docs)
    
    return all_docs

def lsh_cluster_documents(documents, num_hashes=112, num_bands=14, ngram_n=5):
    """
    Applies MinHash + LSH to cluster near-duplicate documents.

    Args:
        documents (List[str]): List of document strings
        num_hashes (int): Length of MinHash signature
        num_bands (int): Number of bands to split signature into
        ngram_n (int): Size of n-grams to use for shingling

    Returns:
        clusters (List[Set[int]]): List of sets of doc IDs (clusters)
        doc_signatures (List[List[int]]): MinHash signatures for each doc
    """
    rows_per_band = num_hashes // num_bands

    buckets = defaultdict(set)  # (band_idx, hash) → doc_ids
    doc_signatures = []

    for doc_id, doc in enumerate(documents):
        n_grams = get_ngrams(doc, n=ngram_n)
        signature = generate_signature(n_grams, num_hashes)
        doc_signatures.append(signature)

        band_hashes = lsh(signature, num_bands, rows_per_band)
        for band_idx, band_hash in enumerate(band_hashes):
            bucket_key = (band_idx, band_hash)
            buckets[bucket_key].add(doc_id)

    # Invert: doc_id → list of buckets
    doc_to_clusters = defaultdict(set)
    for (band_idx, band_hash), doc_ids in buckets.items():
        if len(doc_ids) > 1:
            for doc_id in doc_ids:
                doc_to_clusters[doc_id].add((band_idx, band_hash))

    # Build connected components (clusters)
    visited = set()
    clusters = []

    for doc_id in range(len(documents)):
        if doc_id in visited:
            continue
        cluster = set()
        queue = deque([doc_id])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            cluster.add(current)
            for band_pair in doc_to_clusters[current]:
                queue.extend(buckets[band_pair] - visited)
        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters, doc_signatures


# all_docs = generate_test_documents()
# print(f"Total documents: {len(all_docs)}")
# print(f"Unique documents: {len(set(all_docs))}")
# clusters = lsh_cluster_documents(all_docs)
# for i, cluster in enumerate(clusters):
#     print(f"\n🔗 Cluster {i+1} — {len(cluster)} docs:")
#     for doc_id in sorted(cluster):
#         print(f"  [{doc_id}] {all_docs[doc_id][:80]}...")




#########################################################################
# Visualize the clusters on test data with 5 unique documents and rest repeated



docs = generate_test_documents(4, 20)
print(docs)
clusters, doc_signatures = lsh_cluster_documents(docs)
print("len of clusters with collisions identified: ", len(clusters))
print(type(clusters), clusters[0])

visualize_clusters(clusters=clusters, all_docs=docs)



