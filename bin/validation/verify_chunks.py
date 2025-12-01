#!/usr/bin/env python3
import os
import glob
import torch
import argparse
import random
from tqdm import tqdm
import sys

def verify_chunks(chunks_root, num_samples=None, verbose=False):
    print(f"🔍 Weryfikacja chunków w: {chunks_root}")
    
    # 1. Znajdź wszystkie pliki .pt (rekurencyjnie)
    # Struktura: root/SeqName/chunk_*.pt
    pattern = os.path.join(chunks_root, "**", "chunk_*.pt")
    files = sorted(glob.glob(pattern, recursive=True))
    
    if not files:
        print("❌ Nie znaleziono żadnych plików .pt! Sprawdź ścieżkę.")
        return

    print(f"📦 Znaleziono łącznie {len(files)} plików.")
    
    # 2. Wybór próbki (opcjonalne)
    if num_samples:
        if num_samples > len(files):
            num_samples = len(files)
        files = random.sample(files, num_samples)
        print(f"🔬 Sprawdzam losową próbkę {len(files)} plików...")
    else:
        print("🔬 Sprawdzam WSZYSTKIE pliki (to może chwilę potrwać)...")

    valid_count = 0
    error_count = 0
    features_count = 0
    intra_count = 0
    
    # Statystyki kształtów
    shapes_seen = set()
    
    for fpath in tqdm(files):
        try:
            # Próba załadowania
            data = torch.load(fpath, map_location="cpu")
            
            # Weryfikacja kluczy
            required_keys = ["chunk", "seq_meta", "poc"]
            for k in required_keys:
                if k not in data:
                    raise ValueError(f"Brak klucza: {k}")
            
            # Weryfikacja obrazu
            chunk = data["chunk"]
            if not isinstance(chunk, torch.Tensor):
                 raise TypeError(f"'chunk' nie jest tensorem")
            
            # Oczekiwany kształt: [3, H, W]
            if chunk.ndim != 3 or chunk.shape[0] != 3:
                 raise ValueError(f"Niepoprawny kształt obrazu: {chunk.shape}")
            
            shapes_seen.add(tuple(chunk.shape))
            
            # Weryfikacja metadanych
            if data.get("is_intra", 0) == 1:
                intra_count += 1

            # Weryfikacja map cech (VVC Features)
            if "vvc_features" in data:
                features_count += 1
                feats = data["vvc_features"]
                
                if not isinstance(feats, torch.Tensor):
                    raise TypeError(f"'vvc_features' nie jest tensorem")
                
                # Oczekiwany kształt: [6, H, W]
                if feats.ndim != 3 or feats.shape[0] != 6:
                     raise ValueError(f"Niepoprawny kształt cech: {feats.shape} (oczekiwano 6 kanałów)")
                
                # Spójność wymiarów przestrzennych (H, W)
                if feats.shape[1:] != chunk.shape[1:]:
                     raise ValueError(f"Niezgodność wymiarów: obraz {chunk.shape} vs cechy {feats.shape}")
                
                # Sprawdzenie typu (oczekiwany float16/half)
                if feats.dtype != torch.float16:
                    # To nie błąd krytyczny, ale warto wiedzieć
                    if verbose: print(f" [Info] Cechy mają typ {feats.dtype}, oczekiwano float16")

            valid_count += 1
            
            # Wypisz szczegóły pierwszego poprawnego pliku
            if verbose and valid_count == 1:
                print(f"\n--- Przykładowy plik: {os.path.basename(fpath)} ---")
                print(f" Ścieżka: {fpath}")
                print(f" Klucze: {list(data.keys())}")
                print(f" Chunk (Obraz): {chunk.dtype}, {chunk.shape}, zakres=[{chunk.min()}, {chunk.max()}]")
                if "vvc_features" in data:
                    print(f" Cechy VVC: {data['vvc_features'].dtype}, {data['vvc_features'].shape}")
                else:
                    print(f" Cechy VVC: BRAK (to normalne dla oryginałów)")
                print(f" Metadane: {data['seq_meta']}")
                print("-------------------------------------------\n")

        except Exception as e:
            error_count += 1
            print(f"\n❌ BŁĄD w pliku {fpath}: {e}")
            # Jeśli błędów jest dużo, przerwij
            if error_count > 20:
                print("!!! Zbyt wiele błędów, przerywam weryfikację.")
                break

    print("\n=== Podsumowanie Weryfikacji ===")
    print(f"✅ Poprawne pliki: {valid_count}")
    print(f"✨ Pliki z mapami cech (Fused Maps): {features_count}")
    print(f"🖼️  Pliki typu INTRA: {intra_count}")
    print(f"❌ Uszkodzone pliki: {error_count}")
    print(f"📐 Wykryte rozmiary chunków: {shapes_seen}")
    
    if error_count == 0 and valid_count > 0:
        print("\n🎉 Wygląda to bardzo dobrze! Możesz trenować.")
    else:
        print("\n⚠️  Znaleziono problemy. Sprawdź logi powyżej.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weryfikacja wygenerowanych chunków .pt")
    parser.add_argument("root", help="Katalog z chunkami (np. chunks_pt)")
    parser.add_argument("--sample", type=int, default=None, help="Sprawdź tylko N losowych plików (np. 1000)")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="Pokaż szczegóły")
    args = parser.parse_args()
    
    verify_chunks(args.root, args.sample, args.verbose)