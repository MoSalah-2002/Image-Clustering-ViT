# Image Clustering with Vision Transformer (ViT)

An unsupervised image clustering experiment using pretrained `google/vit-base-patch16-224` to extract image representations, followed by K-Means or Agglomerative Clustering.

## Workflow
- Read images from a directory (JPG, JPEG, PNG, WEBP).
- Extract the ViT CLS-token representation.
- Normalize feature vectors and cluster them.
- Export image names and cluster IDs to CSV.

## Run
`pip install -r requirements.txt`

`python src/cluster_images.py --image-dir ./images --clusters 10 --method agglomerative`

The pretrained model downloads on first run. GPU is recommended for faster extraction.

## Limitations
This is exploratory unsupervised learning with no ground-truth labels. Cluster IDs are arbitrary, and the number of clusters is a user-selected parameter. No semantic-quality benchmark is claimed.