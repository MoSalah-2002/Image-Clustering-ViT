"""Extract ViT CLS features and cluster a directory of images."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTModel
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--method", choices=["kmeans", "agglomerative"], default="agglomerative")
    parser.add_argument("--output", default="clusters.csv")
    args = parser.parse_args()
    paths = sorted(p for p in Path(args.image_dir).iterdir()
                   if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if len(paths) < 2 or not 2 <= args.clusters <= len(paths):
        raise ValueError("Need at least two images and clusters between 2 and image count.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device).eval()
    features = []
    with torch.inference_mode():
        for path in tqdm(paths, desc="Extracting ViT features"):
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            features.append(outputs.last_hidden_state[:, 0, :].cpu().numpy()[0])
    X = normalize(np.vstack(features))
    if args.method == "kmeans":
        labels = KMeans(n_clusters=args.clusters, random_state=42, n_init=10).fit_predict(X)
    else:
        labels = AgglomerativeClustering(n_clusters=args.clusters, metric="cosine",
                                         linkage="average").fit_predict(X)
    result = pd.DataFrame({"image": [p.name for p in paths], "cluster": labels})
    result.to_csv(args.output, index=False)
    print(result.sort_values("cluster").to_string(index=False))
    print(f"Saved assignments to {args.output}")

if __name__ == "__main__":
    main()
