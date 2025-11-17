#!/usr/bin/env python3
"""
Skrypt weryfikujący integralność fused_maps dla wszystkich sekwencji w datasecie.
Sprawdza:
1. Czy folder fused_maps istnieje
2. Czy wszystkie pliki fused_maps_pocX.pt są obecne
3. Czy mają wszystkie 6 kanałów VVC
4. Czy wymiary się zgadzają
"""

import os
import torch
from pathlib import Path
from collections import defaultdict

# Ścieżki
CHUNKS_ROOT = "/mnt/d/data_mgr/chunks_pt"
FUSED_ROOT = "/mnt/d/data_mgr/decoded"

# Oczekiwane kanały VVC
EXPECTED_CHANNELS = [
    "y_ac_energy",
    "y_nz_density", 
    "y_dc",
    "boundary_bin",
    "boundary_weight",
    "size_map_norm"
]

def verify_fused_maps():
    """Weryfikuje fused_maps dla wszystkich sekwencji."""
    
    print("=" * 80)
    print("WERYFIKACJA FUSED_MAPS")
    print("=" * 80)
    print(f"chunks_pt: {CHUNKS_ROOT}")
    print(f"fused maps: {FUSED_ROOT}")
    print()
    
    # Zbierz wszystkie sekwencje z chunks_pt
    sequences = sorted([d for d in os.listdir(CHUNKS_ROOT) 
                       if os.path.isdir(os.path.join(CHUNKS_ROOT, d))])
    
    print(f"Znaleziono {len(sequences)} sekwencji")
    print()
    
    stats = {
        'total_sequences': len(sequences),
        'missing_fused_folder': 0,
        'missing_poc_files': 0,
        'missing_channels': 0,
        'wrong_dimensions': 0,
        'corrupted_files': 0,
        'ok_sequences': 0,
        'total_pocs_checked': 0,
        'total_pocs_missing': 0,
    }
    
    errors = defaultdict(list)
    
    for seq_idx, seq in enumerate(sequences, 1):
        if seq_idx % 100 == 0:
            print(f"[{seq_idx}/{len(sequences)}] Przetwarzanie...")
        
        seq_chunks = os.path.join(CHUNKS_ROOT, seq)
        seq_fused = os.path.join(FUSED_ROOT, seq, "fused_maps")
        
        # 1. Sprawdź czy folder fused_maps istnieje
        if not os.path.isdir(seq_fused):
            stats['missing_fused_folder'] += 1
            errors['missing_folder'].append(seq)
            continue
        
        # 2. Policz POC'y w chunks_pt
        chunk_files = sorted([f for f in os.listdir(seq_chunks) if f.startswith("chunks_poc") and f.endswith(".pt")])
        num_pocs = len(chunk_files)
        stats['total_pocs_checked'] += num_pocs
        
        # 3. Sprawdź pliki fused_maps_pocX.pt
        seq_has_errors = False
        
        for poc_idx in range(num_pocs):
            # Nazwa pliku może być fused_maps_poc0.pt lub fused_maps_poc000.pt
            # Spróbuj obu formatów
            fused_file = None
            for fmt in [f"fused_maps_poc{poc_idx}.pt", f"fused_maps_poc{poc_idx:03d}.pt"]:
                candidate = os.path.join(seq_fused, fmt)
                if os.path.isfile(candidate):
                    fused_file = candidate
                    break
            
            if fused_file is None:
                stats['missing_poc_files'] += 1
                stats['total_pocs_missing'] += 1
                errors['missing_poc'].append(f"{seq}/poc{poc_idx}")
                seq_has_errors = True
                continue
            
            # 4. Wczytaj i sprawdź zawartość
            try:
                data = torch.load(fused_file, map_location='cpu', weights_only=False)
                
                # Sprawdź czy ma wszystkie kanały
                missing_ch = [ch for ch in EXPECTED_CHANNELS if ch not in data]
                if missing_ch:
                    stats['missing_channels'] += 1
                    errors['missing_channels'].append(f"{seq}/poc{poc_idx}: brak {missing_ch}")
                    seq_has_errors = True
                    continue
                
                # Sprawdź wymiary (wszystkie powinny być takie same)
                shapes = {ch: data[ch].shape for ch in EXPECTED_CHANNELS}
                unique_shapes = set(shapes.values())
                if len(unique_shapes) > 1:
                    stats['wrong_dimensions'] += 1
                    errors['wrong_dims'].append(f"{seq}/poc{poc_idx}: różne wymiary {shapes}")
                    seq_has_errors = True
                    
            except Exception as e:
                stats['corrupted_files'] += 1
                errors['corrupted'].append(f"{seq}/poc{poc_idx}: {str(e)}")
                seq_has_errors = True
        
        if not seq_has_errors:
            stats['ok_sequences'] += 1
    
    # Wyświetl podsumowanie
    print()
    print("=" * 80)
    print("PODSUMOWANIE")
    print("=" * 80)
    print(f"Łącznie sekwencji:              {stats['total_sequences']}")
    print(f"Sekwencje OK:                   {stats['ok_sequences']} ({stats['ok_sequences']/stats['total_sequences']*100:.1f}%)")
    print(f"Sekwencje z błędami:            {stats['total_sequences'] - stats['ok_sequences']}")
    print()
    print(f"Brak folderu fused_maps:        {stats['missing_fused_folder']}")
    print(f"Brak plików POC:                {stats['total_pocs_missing']} z {stats['total_pocs_checked']}")
    print(f"Brak kanałów:                   {stats['missing_channels']}")
    print(f"Złe wymiary:                    {stats['wrong_dimensions']}")
    print(f"Uszkodzone pliki:               {stats['corrupted_files']}")
    print()
    
    # Wyświetl przykładowe błędy
    if errors:
        print("=" * 80)
        print("PRZYKŁADOWE BŁĘDY (max 10 na kategorię)")
        print("=" * 80)
        
        for error_type, error_list in errors.items():
            if error_list:
                print(f"\n{error_type.upper()}:")
                for err in error_list[:10]:
                    print(f"  - {err}")
                if len(error_list) > 10:
                    print(f"  ... i {len(error_list) - 10} więcej")
    
    # Status końcowy
    print()
    print("=" * 80)
    if stats['ok_sequences'] == stats['total_sequences']:
        print("✅ WSZYSTKIE FUSED_MAPS SĄ OK!")
    else:
        print(f"⚠️  ZNALEZIONO PROBLEMY W {stats['total_sequences'] - stats['ok_sequences']} SEKWENCJACH")
    print("=" * 80)
    
    return stats, errors

if __name__ == "__main__":
    stats, errors = verify_fused_maps()
