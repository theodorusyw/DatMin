from sklearn.decomposition import PCA
from .data_loader import df
from .constants import BASE_COLS

pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(df[BASE_COLS])
df["PC1"] = pca_components[:, 0]
df["PC2"] = pca_components[:, 1]
PCA_VAR_EXPLAINED = pca.explained_variance_ratio_