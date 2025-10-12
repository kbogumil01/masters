#!/usr/bin/env python3
# quick_test_training.py
# Szybki test uczenia sieci neuronowej z VVC enhancement na małym zbiorze

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import yaml
import tempfile
from torch.utils.data import DataLoader

# Dodaj ścieżki do modułów
sys.path.insert(0, '/home/karol/mgr/new_PC/nn_code')
sys.path.insert(0, '/home/karol/mgr/new_PC/enhancer')

# Import naszych modułów
from enhancer.dataset import VVCDataset
from enhancer.models.enhancer import EnhancerLightning
from enhancer.datamodule import VVCDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def create_test_chunks():
    """Utwórz małe chunks RGB z dostępnych danych YUV dla testu"""
    print("🔧 Creating test RGB chunks from YUV data...")
    
    videos_dir = Path('/home/karol/mgr/new_PC/videos/data')
    test_chunks_dir = Path('/home/karol/mgr/new_PC/test_chunks')
    test_chunks_dir.mkdir(exist_ok=True)
    
    yuv_files = list(videos_dir.glob('*.yuv'))[:2]  # Weź tylko 2 pliki dla testu
    
    for yuv_file in yuv_files:
        print(f"  Processing: {yuv_file.name}")
        
        # Pobierz parametry z .info
        info_file = yuv_file.with_suffix('.yuv.info')
        if not info_file.exists():
            continue
            
        # Parsuj info file
        info = {}
        with open(info_file) as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()
        
        width = int(float(info.get('Width', '352')))
        height = int(float(info.get('Height', '288')))
        
        # Załaduj pierwsze 5 klatek YUV
        frame_size = width * height * 3 // 2  # YUV420
        
        chunks_rgb = []
        chunks_metadata = []
        
        with open(yuv_file, 'rb') as f:
            for frame_idx in range(min(5, 64)):  # Max 5 klatek dla testu
                yuv_data = f.read(frame_size)
                if len(yuv_data) != frame_size:
                    break
                
                # Konwertuj YUV420 do RGB (uproszczone)
                y_size = width * height
                uv_size = y_size // 4
                
                y = np.frombuffer(yuv_data[:y_size], dtype=np.uint8).reshape(height, width)
                u = np.frombuffer(yuv_data[y_size:y_size+uv_size], dtype=np.uint8).reshape(height//2, width//2)
                v = np.frombuffer(yuv_data[y_size+uv_size:], dtype=np.uint8).reshape(height//2, width//2)
                
                # Upsample UV
                u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
                v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)
                
                # Konwersja YUV do RGB (uproszczona)
                rgb = np.stack([
                    np.clip(y + 1.402 * (v - 128), 0, 255),
                    np.clip(y - 0.344 * (u - 128) - 0.714 * (v - 128), 0, 255),
                    np.clip(y + 1.772 * (u - 128), 0, 255)
                ], axis=0).astype(np.uint8)
                
                # Podziel na chunks 132x132
                for y_start in range(0, height-132+1, 132):
                    for x_start in range(0, width-132+1, 132):
                        if y_start + 132 <= height and x_start + 132 <= width:
                            chunk_rgb = rgb[:, y_start:y_start+132, x_start:x_start+132]
                            
                            # Metadata (przykładowe)
                            metadata = np.array([
                                frame_idx / 63.0,  # frame_number_norm
                                x_start / width,    # x_position_norm
                                y_start / height,   # y_position_norm
                                width / 1920.0,     # width_norm
                                height / 1080.0,    # height_norm
                                0.5                  # qp_norm (przykładowe)
                            ], dtype=np.float32)
                            
                            chunks_rgb.append(chunk_rgb)
                            chunks_metadata.append(metadata)
                            
                            # Ogranicz liczbę chunks dla testu
                            if len(chunks_rgb) >= 20:
                                break
                    if len(chunks_rgb) >= 20:
                        break
                if len(chunks_rgb) >= 20:
                    break
        
        # Zapisz chunks
        if chunks_rgb:
            output_file = test_chunks_dir / f"chunks_{yuv_file.stem}.npz"
            np.savez_compressed(
                output_file,
                rgb=np.array(chunks_rgb),
                metadata=np.array(chunks_metadata)
            )
            print(f"  Created: {output_file} ({len(chunks_rgb)} chunks)")
    
    return test_chunks_dir

def create_test_config(chunks_dir, fused_maps_dir):
    """Utwórz konfigurację dla testu"""
    config = {
        'model': {
            'architecture': 'convnet',
            'features': 64,
            'layers': 3,
            'learning_rate': 0.001,
            'use_vvc_features': True  # Włącz VVC enhancement
        },
        'data': {
            'chunks_dir': str(chunks_dir),
            'fused_maps_dir': str(fused_maps_dir),
            'batch_size': 2,  # Mały batch dla testu
            'num_workers': 2,
            'train_split': 0.8,
            'val_split': 0.2
        },
        'training': {
            'max_epochs': 3,  # Tylko 3 epoki dla testu
            'accelerator': 'cpu',  # CPU dla szybkiego testu
            'precision': 32,
            'gradient_clip_val': 1.0
        }
    }
    
    return config

def test_vvc_dataset(chunks_dir, fused_maps_dir):
    """Test czy VVCDataset działa z naszymi danymi"""
    print("🧪 Testing VVCDataset...")
    
    try:
        dataset = VVCDataset(
            chunks_dir=chunks_dir,
            fused_maps_dir=fused_maps_dir,
            split='train',
            train_split=0.8
        )
        
        print(f"  Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test pojedynczego sampla
            rgb, orig_rgb, metadata, chunk_obj, vvc_features = dataset[0]
            
            print(f"  RGB shape: {rgb.shape}")
            print(f"  Orig RGB shape: {orig_rgb.shape}")
            print(f"  Metadata shape: {metadata.shape}")
            print(f"  VVC features shape: {vvc_features.shape}")
            print(f"  Chunk object: {chunk_obj}")
            
            return True
        else:
            print("  ❌ Dataset is empty!")
            return False
            
    except Exception as e:
        print(f"  ❌ Dataset test failed: {e}")
        return False

def test_model_creation():
    """Test czy model się tworzy poprawnie"""
    print("🧪 Testing model creation...")
    
    try:
        config = {
            'architecture': 'convnet',
            'features': 64,
            'layers': 3,
            'learning_rate': 0.001,
            'use_vvc_features': True
        }
        
        model = EnhancerLightning(config)
        print(f"  Model created successfully")
        print(f"  Model type: {type(model.net).__name__}")
        
        # Test forward pass
        batch_size = 2
        rgb = torch.randn(batch_size, 3, 132, 132)
        metadata = torch.randn(batch_size, 6)
        vvc_features = torch.randn(batch_size, 13, 132, 132)
        
        with torch.no_grad():
            output = model.net(rgb, metadata, vvc_features)
            print(f"  Forward pass successful, output shape: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        return False

def run_quick_training(config):
    """Uruchom szybkie uczenie dla testu"""
    print("🚀 Starting quick training test...")
    
    try:
        # Utwórz data module
        datamodule = VVCDataModule(
            chunks_dir=config['data']['chunks_dir'],
            fused_maps_dir=config['data']['fused_maps_dir'],
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            train_split=config['data']['train_split'],
            val_split=config['data']['val_split']
        )
        
        # Utwórz model
        model = EnhancerLightning(config['model'])
        
        # Utwórz trainer
        trainer = pl.Trainer(
            max_epochs=config['training']['max_epochs'],
            accelerator=config['training']['accelerator'],
            precision=config['training']['precision'],
            gradient_clip_val=config['training']['gradient_clip_val'],
            enable_checkpointing=False,  # Bez checkpointów dla testu
            logger=False,  # Bez loggera dla testu
            enable_progress_bar=True,
            num_sanity_val_steps=1  # Tylko 1 validation step dla testu
        )
        
        # Uruchom uczenie
        trainer.fit(model, datamodule)
        
        print("✅ Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== VVC ENHANCED NEURAL NETWORK - QUICK TEST ===")
    
    # 1. Sprawdź dostępne fused maps
    videos_decoded_dir = Path('/home/karol/mgr/new_PC/videos/decoded')
    fused_maps_dirs = list(videos_decoded_dir.glob('*/fused_maps'))
    
    if not fused_maps_dirs:
        print("❌ No fused maps found in videos/decoded/")
        return False
    
    print(f"Found {len(fused_maps_dirs)} fused maps directories")
    fused_maps_dir = fused_maps_dirs[0].parent  # Pierwszy katalog decoded
    print(f"Using: {fused_maps_dir}")
    
    # 2. Utwórz test chunks
    chunks_dir = create_test_chunks()
    
    # 3. Test dataset
    if not test_vvc_dataset(chunks_dir, fused_maps_dir):
        return False
    
    # 4. Test model
    if not test_model_creation():
        return False
    
    # 5. Utwórz konfigurację
    config = create_test_config(chunks_dir, fused_maps_dir)
    
    # 6. Uruchom szybkie uczenie
    if run_quick_training(config):
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ VVC Enhanced Neural Network is working correctly!")
        return True
    else:
        print("\n❌ Training test failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)