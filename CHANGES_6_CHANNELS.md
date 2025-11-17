# Zmiany: Przejście z 13 na 6 kanałów VVC + format .pt

**Data:** 2025-11-12

## Podsumowanie zmian

Projekt został zaktualizowany do używania **6 kanałów VVC** zamiast 13 oraz formatu **`.pt`** zamiast `.npz`.

### Zmienione kanały VVC:

**Poprzednio (13 kanałów):**
- Dequant maps: y_ac_energy, y_nz_density, y_dc (3)
- Boundary maps: boundary_bin, boundary_weight, size_map_norm (3)
- Enhanced features: block_energy_contrast, quantization_severity, boundary_energy_drop, block_size_category, complexity_mismatch, ac_density_per_block, dc_variation (7)

**Aktualnie (6 kanałów):**
- Dequant maps: y_ac_energy, y_nz_density, y_dc (3)
- Boundary maps: boundary_bin, boundary_weight, size_map_norm (3)
- ❌ Usunięto: wszystkie 7 enhanced features (były obliczane w fuse_maps.py ale nieużywane)

### Format danych:

- **Chunki**: `.pt` (poprzednio: `.npz`)
  - Generowane przez: `bin/split_to_chunks_pt.py`
  - Lokalizacja: `chunks_pt/` i `orig_chunks_pt/`

- **Fused maps**: `.pt` (poprzednio: `.npz`)
  - Generowane przez: `bin/fuse_maps.py`
  - Lokalizacja: `decoded/<seq>/fused_maps/fused_maps_poc{N}.pt`

## Zmienione pliki:

### 1. **enhancer/models/enhancer.py**
- `VVCFeatureEncoder.__init__()`: `in_channels=13` → `in_channels=6`
- `VVCFeatureEncoder.forward()`: Zaktualizowana dokumentacja (B, 6, H, W)
- `Enhancer.__init__()`: Zaktualizowany komentarz dla VVCFeatureEncoder

### 2. **enhancer/dataset_pt.py**
- Zaktualizowana dokumentacja klasy `VVCChunksPTDataset`
- Dodano opis 6 kanałów VVC

### 3. **enhancer/dataset_npz.py** (legacy support)
- `__init__()`: `torch.zeros(13, ...)` → `torch.zeros(6, ...)`
- `_load_fused_maps()`: Usunięto 7 enhanced features z `channel_keys`
- Wszystkie error handlers: `torch.zeros(13, ...)` → `torch.zeros(6, ...)`
- Zaktualizowane komentarze: "13 channels" → "6 channels"

### 4. **enhancer/dataset.py** (legacy support)
- `_load_vvc_features()`: Usunięto 7 enhanced features z `map_names`
- Wszystkie `torch.zeros(13, ...)` → `torch.zeros(6, ...)`
- Zaktualizowane komentarze

### 5. **enhancer/datamodule.py**
- ✅ Bez zmian - automatyczna detekcja .pt/.npz przez `get_vvc_dataset()`

### 6. **bin/fuse_maps.py**
- ✅ Już zaktualizowany - zapisuje tylko 6 kanałów w `.pt`
- Linie 343-368: Zapisuje y_ac_energy, y_nz_density, y_dc, boundary_bin, boundary_weight, size_map_norm

### 7. **bin/split_to_chunks_pt.py**
- ✅ Gotowy do użycia - generuje `.pt` chunks

## Pipeline przygotowania danych:

```bash
# 1. Dekodowanie VVC (generuje recon.yuv + mapy)
bin/decode_data.sh videos/encoded/<seq>.vvc videos/decoded/

# 2. Generowanie fused maps (.pt, 6 kanałów)
bin/fuse_maps.py --input-dir decoded/<seq>/ --outdir decoded/<seq>/fused_maps/

# 3. Split do chunków (.pt)
bin/split_to_chunks_pt.py decoded/ data/ chunks_pt/      # decoded chunks
bin/split_to_chunks_pt.py data/ data/ orig_chunks_pt/    # original chunks

# 4. Trening
python -m enhancer --config experiments/<config>.yaml
```

## Architektura modelu:

```
Input: 
  - RGB: 3 kanały (decoded frame)
  - Metadata: 4 wartości (QP, ALF, SAO, DB) → 4 kanały po interpolacji
  - VVC features: 6 kanałów → 16 kanałów po VVCFeatureEncoder

Total input to backbone: 3 + 4 + 16 = 23 kanały
```

## Kompatybilność wsteczna:

- ✅ Stary kod .npz nadal działa (dataset.py, dataset_npz.py)
- ✅ Automatyczna detekcja formatu przez `get_vvc_dataset()`
- ✅ Fallback do zer jeśli brak VVC features

## Testowanie:

```python
# Test VVCFeatureEncoder
from enhancer.models.enhancer import VVCFeatureEncoder
import torch

encoder = VVCFeatureEncoder(in_channels=6, out_channels=16)
x = torch.randn(2, 6, 132, 132)  # batch=2, 6 channels, 132x132
out = encoder(x)
print(out.shape)  # torch.Size([2, 16, 132, 132])
```

## Weryfikacja przed treningiem:

1. ✅ VVCFeatureEncoder akceptuje 6 kanałów
2. ✅ Dataset zwraca tensory (B, 6, H, W)
3. ✅ Fused maps mają 6 kluczy w .pt
4. ✅ Wszystkie fallbacki używają torch.zeros(6, ...)

## Potencjalne problemy:

- ⚠️ Stare checkpointy modelu (.ckpt) z 13 kanałami **NIE ZADZIAŁAJĄ** - trzeba trenować od nowa
- ⚠️ Upewnij się, że `fused_maps/*.pt` mają wszystkie 6 kluczy
- ⚠️ Dataset .npz z 13 kanałami będzie próbował załadować tylko 6 (może być warning)

## Korzyści:

- ✅ **Mniejsze zużycie RAM/VRAM**: 6 kanałów zamiast 13
- ✅ **Szybsze ładowanie**: .pt jest natywne dla PyTorch
- ✅ **Prostszy kod**: mniej enhanced features do utrzymania
- ✅ **Lepsza kompatybilność**: torch.save/load lepsze niż np.savez

---
**Autor zmian:** AI Assistant  
**Przeglądnięte pliki:** enhancer/{models/enhancer.py, dataset_pt.py, dataset_npz.py, dataset.py, datamodule.py}
