import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from config_new import PROC_EMBED_DIR, N_SEC_PCA

# Load manifest
manifest_path = PROC_EMBED_DIR / "manifest.csv"
manifest_df = pd.read_csv(manifest_path, names=['ticker', 'date', 'path', 'dim'])

# Load all embeddings
all_embeddings = []
for idx, row in manifest_df.iterrows():
    emb_path = Path(row['path'])
    if not emb_path.is_absolute():
        emb_path = PROC_EMBED_DIR / emb_path
    
    if emb_path.exists():
        embedding = np.load(emb_path)
        all_embeddings.append(embedding)
        print(f"Loaded {emb_path.name}, shape: {embedding.shape}")

# Stack embeddings
X = np.vstack(all_embeddings)
print(f"Total embeddings: {X.shape[0]}, Dimension: {X.shape[1]}")

# Apply PCA
n_components = min(N_SEC_PCA, X.shape[0], X.shape[1])
if n_components < N_SEC_PCA:
    print(f"Warning: reducing PCA components from {N_SEC_PCA} to {n_components} because only {X.shape[0]} samples are available.")
pca = PCA(n_components=n_components)
pca.fit(X)
print(f"PCA fitted. Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Save PCA model
pca_path = PROC_EMBED_DIR / "sec_pca.joblib"
joblib.dump(pca, pca_path)
print(f"Saved PCA model to {pca_path}")

# Transform and save PCA features
for idx, row in manifest_df.iterrows():
    emb_path = Path(row['path'])
    if not emb_path.is_absolute():
        emb_path = PROC_EMBED_DIR / emb_path
    
    if emb_path.exists():
        embedding = np.load(emb_path)
        pca_features = pca.transform(embedding.reshape(1, -1))[0]
        
        # Save PCA features
        pca_path_file = emb_path.with_name(emb_path.stem + "_pca.npy")
        np.save(pca_path_file, pca_features)
        print(f"Saved PCA features to {pca_path_file}")

print("Done!")