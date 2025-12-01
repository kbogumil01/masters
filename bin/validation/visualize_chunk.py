#!/usr/bin/env python3
import argparse
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Nazwy kanałów cech zgodnie z kolejnością w split_to_chunks_pt.py
FEATURE_NAMES = [
    "AC Energy (Energia)",
    "NZ Density (Gęstość)",
    "DC Coeff (Składowa stała)",
    "Boundary Bin (Krawędzie)",
    "Boundary Weight (Wagi)",
    "Size Map (Rozmiar bloku)"
]

def visualize_chunk(chunk_path, output_path):
    if not os.path.exists(chunk_path):
        print(f"❌ Nie znaleziono pliku: {chunk_path}")
        return

    # Ładowanie
    print(f"Loading {chunk_path}...")
    data = torch.load(chunk_path, map_location="cpu")
    
    chunk = data["chunk"]            # [3, H, W], uint8 (YUV)
    features = data.get("vvc_features") # [6, H, W], float16 (lub None)
    
    # Konwersja tensora na numpy: CHW -> HWC
    img_yuv = chunk.permute(1, 2, 0).numpy().astype(np.uint8)
    
    # === NAPRAWA KOLORÓW ===
    # Dane w chunku to YUV. Matplotlib oczekuje RGB.
    img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # Przygotowanie figury
    if features is not None:
        feats = features.float().numpy()
        num_feats = feats.shape[0]
        
        # Układ: Góra (Obraz), Dół (2 rzędy po 3 mapy)
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 1, 1])
        
        # 1. Obraz Główny (zdekodowany chunk)
        ax_img = fig.add_subplot(gs[0, 1]) # Środek góry
        ax_img.imshow(img_rgb)
        ax_img.set_title(f"Chunk Image (RGB)\n{os.path.basename(chunk_path)}", fontsize=12, fontweight='bold')
        ax_img.axis("off")
        
        # Dodatkowe info tekstowe po lewej
        ax_info = fig.add_subplot(gs[0, 0])
        ax_info.axis("off")
        info_text = (
            f"POC: {data['poc']}\n"
            f"QP: {data['seq_meta']['qp']}\n"
            f"Profile: {data['seq_meta']['profile']}\n"
            f"Intra: {data['is_intra']}\n"
            f"Coords: {data.get('coords', '?')}"
        )
        ax_info.text(0.1, 0.5, info_text, fontsize=12, va='center', family='monospace')

        # 2. Mapy cech
        for i in range(num_feats):
            row = 1 + (i // 3)
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            
            f_map = feats[i]
            
            # Dobór colormap
            cmap = 'viridis'
            if "Boundary" in FEATURE_NAMES[i]: cmap = 'magma'
            if "DC" in FEATURE_NAMES[i]: cmap = 'coolwarm'
            
            im = ax.imshow(f_map, cmap=cmap)
            ax.set_title(FEATURE_NAMES[i])
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
    else:
        # Tylko obraz (dla oryginałów)
        plt.figure(figsize=(6, 6))
        plt.imshow(img_rgb)
        plt.title(f"Original Chunk (No Features)\n{os.path.basename(chunk_path)}")
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Zapisano wizualizację: {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_path", help="Ścieżka do pliku .pt z chunkiem")
    parser.add_argument("--output", "-o", default="chunk_viz.png", help="Plik wynikowy PNG")
    args = parser.parse_args()
    
    try:
        import matplotlib
    except ImportError:
        print("⚠️ Brakuje biblioteki matplotlib. Zainstaluj ją: pip install matplotlib")
        exit(1)

    visualize_chunk(args.chunk_path, args.output)