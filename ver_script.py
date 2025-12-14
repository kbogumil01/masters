import torch
import matplotlib.pyplot as plt
import os
import sys

# Ścieżki z Twojego configu
compressed_root = "videos_test/test_frames_pt"
orig_root = "videos_test/test_orig_frames_pt"

def verify_data():
    # 1. Znajdź folder z QP37/42
    if not os.path.exists(compressed_root):
        print(f"Błąd: Nie znaleziono folderu {compressed_root}")
        return

    subdirs = sorted([d for d in os.listdir(compressed_root) if os.path.isdir(os.path.join(compressed_root, d))])
    
    target_dir_name = None
    for d in subdirs:
        if "QP37" in d or "QP42" in d:
            target_dir_name = d
            break
    
    if not target_dir_name:
        if not subdirs: return
        target_dir_name = subdirs[0]

    print(f"Sprawdzam sekwencję: {target_dir_name}")
    comp_dir_path = os.path.join(compressed_root, target_dir_name)

    # 2. Pobierz plik klatki
    frame_files = sorted([f for f in os.listdir(comp_dir_path) if f.endswith('.pt')])
    if not frame_files:
        print(f"Błąd: Pusty folder {target_dir_name}")
        return

    test_frame_file = frame_files[0]
    print(f"Analizuję plik: {test_frame_file}")

    # 3. Ustal ścieżkę oryginału
    if "_AI_" in target_dir_name:
        seq_name = target_dir_name.split("_AI_")[0]
    elif "_RA_" in target_dir_name:
        seq_name = target_dir_name.split("_RA_")[0]
    else:
        seq_name = target_dir_name 

    orig_frame_path = os.path.join(orig_root, seq_name, test_frame_file)
    if not os.path.exists(orig_frame_path):
         flat_name = f"{seq_name}_{test_frame_file}"
         orig_frame_path = os.path.join(orig_root, flat_name)

    # 4. Ładowanie
    try:
        comp_obj = torch.load(os.path.join(comp_dir_path, test_frame_file))
        orig_obj = torch.load(orig_frame_path)
    except Exception as e:
        print(f"Błąd ładowania: {e}")
        return

    # --- NOWA LOGIKA: Wyciąganie 'chunk' ---
    def extract_tensor(obj, name="plik"):
        if isinstance(obj, torch.Tensor):
            return obj
        elif isinstance(obj, dict):
            # Tutaj jest poprawka - szukamy 'chunk' w pierwszej kolejności
            target_key = 'chunk'
            if target_key in obj:
                print(f"INFO: Znaleziono tensor pod kluczem '{target_key}' w {name}.")
                return obj[target_key]
            
            # Fallback
            for key in ['data', 'image', 'frame', 'tensor']:
                if key in obj:
                    return obj[key]
            
            print(f"BŁĄD: Słownik {name} ma klucze {list(obj.keys())}, ale nie ma 'chunk' ani obrazu.")
            return None
        else:
            print(f"BŁĄD: Nieznany typ danych w {name}: {type(obj)}")
            return None

    comp_tensor = extract_tensor(comp_obj, "Skompresowany")
    orig_tensor = extract_tensor(orig_obj, "Oryginał")

    if comp_tensor is None or orig_tensor is None:
        return

    # Upewnienie się co do typów
    if hasattr(comp_tensor, 'is_cuda') and comp_tensor.is_cuda: comp_tensor = comp_tensor.cpu()
    if hasattr(orig_tensor, 'is_cuda') and orig_tensor.is_cuda: orig_tensor = orig_tensor.cpu()
    
    comp_tensor = comp_tensor.float()
    orig_tensor = orig_tensor.float()

    # Normalizacja [0, 255] -> [0, 1]
    if comp_tensor.max() > 1.05: comp_tensor /= 255.0
    if orig_tensor.max() > 1.05: orig_tensor /= 255.0

    # 5. Obliczenia
    mse = torch.mean((comp_tensor - orig_tensor) ** 2)
    
    print("\n=== WYNIK DIAGNOSTYKI ===")
    if mse == 0:
        print("ALARM: Pliki są IDENTYCZNE! (MSE = 0.0)")
        print("-> Twój zbiór treningowy zawiera ORYGINAŁY zamiast skompresowanych klatek.")
    else:
        psnr = 10 * torch.log10(1 / mse)
        print(f"MSE: {mse:.6f}")
        print(f"PSNR: {psnr:.2f} dB")
        
        if psnr > 40:
            print("DIAGNOZA: PSNR > 40 dB. To jest za wysoka jakość jak na QP37.")
            print("-> Prawdopodobnie skrypt generujący .pt wziął złe pliki źródłowe lub VVC nie skompresowało ich poprawnie.")
        else:
            print("DIAGNOZA: PSNR ~30-38 dB. Dane wyglądają na poprawne.")
            print("-> Jeśli tutaj jest OK, a w treningu masz 47dB, problem jest w DataLoaderze.")

if __name__ == "__main__":
    verify_data()