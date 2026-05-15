"""05_create_pca.py
Fit PCA on SEC embeddings (training period only) and save PCA model + transformed features.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from config_new import BASE_DIR, PROC_EMBED_DIR, N_SEC_PCA, TRAIN_SPLIT_DATE


def resolve_embedding_path(path_value):
    path_text = str(path_value).replace("\\", "/")
    path = Path(path_text)

    if path.is_absolute() or (path.parts and ":" in path.parts[0]):
        return path

    if len(path.parts) > 1:
        return BASE_DIR / path

    return PROC_EMBED_DIR / path.name

# Load manifest
manifest_path = PROC_EMBED_DIR / "manifest.csv"
manifest_df = pd.read_csv(manifest_path, names=['ticker', 'date', 'path', 'dim'])
manifest_df['date'] = pd.to_datetime(manifest_df['date'], errors='coerce', format='mixed', dayfirst=True)

# Load all embeddings
# Fit PCA only on training-period embeddings to avoid look-ahead
train_manifest = manifest_df[manifest_df['date'] < pd.to_datetime(TRAIN_SPLIT_DATE)]
all_embeddings = []
for idx, row in train_manifest.iterrows():
    emb_path = resolve_embedding_path(row['path'])
    if emb_path.exists():
        embedding = np.load(emb_path)
        all_embeddings.append(embedding)
        print(f"Loaded train {emb_path.name}, shape: {embedding.shape}")

if len(all_embeddings) == 0:
    raise RuntimeError(f"No training embeddings found before TRAIN_SPLIT_DATE={TRAIN_SPLIT_DATE}")

# Stack embeddings
X = np.vstack(all_embeddings)
print(f"Training embeddings used for PCA fit: {X.shape[0]}, Dimension: {X.shape[1]}")

# Apply PCA
n_components = min(N_SEC_PCA, X.shape[0], X.shape[1])
if n_components < N_SEC_PCA:
    print(f"Warning: reducing PCA components from {N_SEC_PCA} to {n_components} because only {X.shape[0]} samples are available.")
pca = PCA(n_components=n_components)
pca.fit(X)
print(f"PCA fitted on training embeddings. Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Save PCA model
pca_path = PROC_EMBED_DIR / "sec_pca.joblib"
joblib.dump(pca, pca_path)
print(f"Saved PCA model to {pca_path}")

# Transform and save PCA features
# Transform and save PCA features for all embeddings (train + test)
for idx, row in manifest_df.iterrows():
    emb_path = resolve_embedding_path(row['path'])
    if emb_path.exists():
        embedding = np.load(emb_path)
        pca_features = pca.transform(embedding.reshape(1, -1))[0]

        # Save PCA features
        pca_path_file = emb_path.with_name(emb_path.stem + "_pca.npy")
        np.save(pca_path_file, pca_features)
        print(f"Saved PCA features to {pca_path_file}")

print("Done!")