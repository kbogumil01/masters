#!/usr/bin/env python3
import os
import glob
import time
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import psutil

class BenchmarkDataset(Dataset):
    def __init__(self, root_dir, limit=None):
        self.root_dir = root_dir
        print(f"🔍 Skanowanie plików w {root_dir}...")
        # Szukamy plików .pt (takich jak Twoje chunki)
        self.files = sorted(glob.glob(os.path.join(root_dir, "**", "*.pt"), recursive=True))
        
        if not self.files:
            # Fallback dla testów jeśli nie ma .pt - szukamy czegokolwiek
            self.files = sorted(glob.glob(os.path.join(root_dir, "**", "*.*"), recursive=True))
            
        if limit:
            self.files = self.files[:limit]
            
        self.total_size_mb = sum(os.path.getsize(f) for f in self.files) / (1024 * 1024)
        print(f"📦 Znaleziono {len(self.files)} plików (Łącznie: {self.total_size_mb:.2f} MB)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        # Symulacja realnego obciążenia: torch.load (I/O + deserializacja CPU)
        try:
            data = torch.load(path, map_location="cpu")
            return 1 # Dummy return
        except Exception:
            # Fallback dla plików niebędących tensorami (np. zwykły odczyt)
            with open(path, 'rb') as f:
                _ = f.read()
            return 1

def drop_caches():
    """
    Próba wyczyszczenia cache systemu plików (wymaga sudo).
    W WSL może nie działać bez uprawnień, ale warto spróbować symulacji
    poprzez alokację dużej pamięci.
    """
    print("🧹 Próba wyczyszczenia buforów RAM (aby testować dysk, a nie RAM)...")
    try:
        # Alokacja dużej tablicy, żeby wymusić wymianę pamięci (prymitywne, ale działa w user-space)
        _ = [0] * (1024 * 1024 * 100) # ~800MB śmieci
    except:
        pass

def run_benchmark(name, path, batch_size, workers, limit):
    print(f"\n{'='*10} TEST: {name} {'='*10}")
    print(f"📂 Ścieżka: {path}")
    
    if not os.path.exists(path):
        print(f"❌ Ścieżka nie istnieje: {path}")
        return None

    # Wymuszamy czyszczenie cache przed każdym testem
    drop_caches()

    dataset = BenchmarkDataset(path, limit=limit)
    if len(dataset) == 0:
        print("❌ Pusty dataset.")
        return None

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=workers, 
        shuffle=True, # Shuffle ważny, żeby testować random read (jak w treningu)
        pin_memory=True
    )

    print(f"🚀 Start benchmarku (Batch: {batch_size}, Workers: {workers})...")
    
    start_time = time.time()
    count = 0
    
    # Pętla symulująca epokę treningową
    for _ in tqdm(loader, desc=f"Reading from {name}", unit="batch"):
        count += batch_size
        
    end_time = time.time()
    duration = end_time - start_time
    
    throughput = len(dataset) / duration
    bandwidth = dataset.total_size_mb / duration
    
    print(f"\n📊 WYNIKI DLA {name}:")
    print(f"   Czas trwania: {duration:.2f} s")
    print(f"   Przepustowość (ilość): {throughput:.2f} plików/s")
    print(f"   Przepustowość (dane):  {bandwidth:.2f} MB/s")
    
    return throughput

def main():
    parser = argparse.ArgumentParser(description="Benchmark I/O dla DataLoader")
    parser.add_argument("--nvme", help="Ścieżka do folderu na dysku NVMe (np. chunks_pt)", required=True)
    parser.add_argument("--usb", help="Ścieżka do folderu na dysku USB (np. fused_maps lub kopia chunks)", required=True)
    parser.add_argument("--limit", type=int, default=5000, help="Liczba plików do przetestowania (domyślnie 5000)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (taki jak w treningu)")
    parser.add_argument("--workers", type=int, default=4, help="Liczba workerów (taka jak w treningu)")
    
    args = parser.parse_args()
    
    print(f"🖥️  CPU Cores: {psutil.cpu_count(logical=True)}")
    print(f"💾 RAM Available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # Test NVMe
    speed_nvme = run_benchmark("NVMe (Dysk C/WSL)", args.nvme, args.batch, args.workers, args.limit)
    
    # Test USB
    speed_usb = run_benchmark("USB (Zewnętrzny)", args.usb, args.batch, args.workers, args.limit)
    
    if speed_nvme and speed_usb:
        ratio = speed_nvme / speed_usb
        print(f"\n{'='*30}")
        print(f"🏆 PODSUMOWANIE:")
        print(f"NVMe jest {ratio:.2f}x szybsze od USB w tym zadaniu.")
        
        if ratio > 1.5:
            print("⚠️  USB jest znaczącym wąskim gardłem. Zalecane przeniesienie chunków na NVMe.")
        else:
            print("✅ USB radzi sobie nieźle (prawdopodobnie cache systemu operacyjnego pomaga).")

if __name__ == "__main__":
    main()