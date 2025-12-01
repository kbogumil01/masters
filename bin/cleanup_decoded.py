#!/usr/bin/env python3
import os
import glob
import shutil
import argparse
from tqdm import tqdm

def cleanup(decoded_root, chunks_root, min_chunks=100, delete=False):
    """
    Usuwa katalogi z decoded_root, jeśli ich odpowiedniki w chunks_root
    mają więcej niż min_chunks plików .pt.
    """
    # Pobieramy listę folderów w decoded
    decoded_dirs = sorted([d for d in os.listdir(decoded_root) if os.path.isdir(os.path.join(decoded_root, d))])
    
    print(f"📂 Znaleziono {len(decoded_dirs)} folderów w {decoded_root}")
    print(f"🎯 Cel (chunks): {chunks_root}")
    print(f"⚠️  TRYB: {'KASOWANIE (DELETE)' if delete else 'SYMULACJA (DRY RUN)'}")
    print("-" * 40)

    deleted_count = 0
    skipped_count = 0
    space_saved = 0

    for seq_name in tqdm(decoded_dirs):
        dec_path = os.path.join(decoded_root, seq_name)
        chunk_path = os.path.join(chunks_root, seq_name)

        # 1. Sprawdź czy folder z chunkami w ogóle istnieje
        if not os.path.isdir(chunk_path):
            # print(f"SKIP: {seq_name} (brak w chunks)")
            skipped_count += 1
            continue

        # 2. Sprawdź czy ma pliki (czy przetwarzanie się powiodło)
        # Szybkie sprawdzenie liczby plików .pt
        pt_files = glob.glob(os.path.join(chunk_path, "chunk_*.pt"))
        num_chunks = len(pt_files)

        if num_chunks < min_chunks:
            print(f"⚠️  SKIP: {seq_name} (za mało chunków: {num_chunks}, możliwe przerwanie)")
            skipped_count += 1
            continue

        # 3. Bezpieczne usuwanie
        # Obliczanie rozmiaru (tylko dla statystyki)
        total_size = 0
        if delete:
            for dirpath, _, filenames in os.walk(dec_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            
            try:
                shutil.rmtree(dec_path)
                space_saved += total_size
                # print(f"✅ USUNIĘTO: {seq_name}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ BŁĄD przy usuwaniu {seq_name}: {e}")
        else:
            # W trybie dry run tylko logujemy
            # print(f"DO USUNIĘCIA: {seq_name} (ma {num_chunks} chunków)")
            deleted_count += 1

    print("-" * 40)
    if delete:
        print(f"🗑️  Usunięto {deleted_count} folderów.")
        print(f"💾 Zwolniono ok. {space_saved / (1024**3):.2f} GB")
    else:
        print(f"🔍 Znaleziono {deleted_count} folderów kwalifikujących się do usunięcia.")
        print(f"Aby je usunąć, uruchom skrypt z flagą: --delete")
    
    print(f"⏭️  Pominięto {skipped_count} folderów (nieprzetworzone lub niepełne).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Usuwa przetworzone sekwencje z dysku źródłowego.")
    parser.add_argument("decoded_root", help="Folder źródłowy (do czyszczenia), np. /mnt/d/data_mgr/decoded")
    parser.add_argument("chunks_root", help="Folder docelowy (do weryfikacji), np. /mnt/d/data_mgr/chunks_pt")
    parser.add_argument("--min-chunks", type=int, default=50, help="Minimalna liczba chunków, aby uznać sekwencję za gotową")
    parser.add_argument("--delete", action="store_true", help="Wykonaj faktyczne usuwanie (bez tej flagi tylko symulacja)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.decoded_root):
        print(f"Błąd: Nie znaleziono {args.decoded_root}")
        exit(1)

    cleanup(args.decoded_root, args.chunks_root, args.min_chunks, args.delete)