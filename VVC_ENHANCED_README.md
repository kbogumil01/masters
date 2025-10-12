# VVC-Enhanced Neural Network Integration

## 🚀 Overview

This integration extends the original neural network with **VVC intelligence** from decoded video data. The network now processes:

- **RGB chunks** (3 channels, 132×132) - original video content
- **Metadata** (6 features) - encoding parameters (QP, profile, etc.)
- **VVC features** (13 channels) - codec intelligence from fused maps

**Total input: 25 channels** instead of the original 9 channels.

## 📊 VVC Intelligence Components

The 13 VVC intelligence channels include:

### Original Dequant Maps (3 channels):
- `y_ac_energy` - AC coefficient energy distribution
- `y_nz_density` - Non-zero coefficient density
- `y_dc` - DC coefficient values

### Boundary Maps (3 channels):
- `boundary_bin` - Binary block boundaries
- `boundary_weight` - Weighted boundary strength
- `size_map_norm` - Normalized block size information

### Enhanced Features (7 channels):
- `block_energy_contrast` - Energy variation within blocks
- `quantization_severity` - Estimated quantization loss
- `boundary_energy_drop` - Energy loss at block boundaries
- `block_size_category` - Classified block sizes (0-3)
- `complexity_mismatch` - Size vs complexity inconsistencies
- `ac_density_per_block` - Block-normalized AC density
- `dc_variation` - DC coefficient variation within blocks

## 🔧 Usage

### 1. Generate VVC Intelligence Data

First, decode your VVC bitstreams with intelligence extraction:

```bash
# This generates fused_maps/fused_maps_poc*.npz files
bin/decode_data.sh video.vvc output_dir/
```

### 2. Configure the Enhanced Network

Create a configuration file (see `vvc_enhanced_config.yaml`):

```yaml
# Point to your fused maps directory
fused_maps_dir: "path/to/output_dir/fused_maps"

# Ensure output_block is configured for 3 RGB channels
enhancer:
  output_block:
    features: 3
    tanh: false
```

### 3. Train with VVC Intelligence

```python
from enhancer.datamodule import VVCDataModule
from enhancer.trainer_module import LitEnhancer
from enhancer.config import Config

# Load configuration
config = Config.from_yaml("vvc_enhanced_config.yaml")

# Create datamodule with VVC features
datamodule = VVCDataModule(
    dataset_config=config.dataset,
    dataloader_config=config.dataloader,
    fused_maps_dir=config.fused_maps_dir  # NEW
)

# Create trainer
trainer = LitEnhancer(config.trainer)

# Train with VVC intelligence!
trainer.fit(model, datamodule)
```

## 🏗️ Architecture Details

### VVC Feature Processing Pipeline

```
VVC Maps (13, 132, 132)
    ↓
VVCFeatureEncoder
    ├── Conv2d(13→32, 3×3)
    ├── BatchNorm + ReLU
    ├── Conv2d(32→16, 3×3)
    └── BatchNorm + ReLU
    ↓
Processed VVC (16, 132, 132)
```

### Complete Network Flow

```
RGB (3, 132, 132) ────┐
                      ├── Concatenate → (25, 132, 132)
Metadata (6) ─────────┤      ↓
(expanded to spatial) │   Main Network
                      │   (Conv/Dense/Res)
VVC (13, 132, 132) ───┤      ↓
(processed to 16)     │   Output (3, 132, 132)
                      │      ↓
Original RGB ─────────┴── Add (if with_mask)
                           ↓
                    Enhanced RGB (3, 132, 132)
```

## 📁 File Structure

### Modified Files:
- `enhancer/dataset.py` - Added VVC feature loading
- `enhancer/models/enhancer.py` - Added VVCFeatureEncoder
- `enhancer/datamodule.py` - Added fused_maps_dir parameter  
- `enhancer/trainer_module.py` - Updated for 5-tensor batches

### New Files:
- `test_vvc_enhanced.py` - Integration test
- `vvc_enhanced_config.yaml` - Example configuration

### Data Flow:
```
decode_data.sh → fused_maps/fused_maps_poc*.npz
                        ↓
VVCDataset._load_vvc_features() → VVC tensors (13, 132, 132)
                        ↓
VVCFeatureEncoder → Processed VVC (16, 132, 132)
                        ↓
Enhanced Neural Network → Better RGB enhancement
```

## 🔄 Backward Compatibility

The enhanced network maintains full backward compatibility:

- **With VVC features**: Uses full 25-channel processing
- **Without VVC features**: Automatically fills with zeros, behaves like original 9-channel network
- **Existing configs**: Work without modification (VVC features disabled)

## 🧪 Testing

```bash
# Test the integration
python test_vvc_enhanced.py --fused-maps-dir path/to/fused_maps

# Expected output:
# 🚀 VVC-ENHANCED NEURAL NETWORK INTEGRATION TEST
# ✅ VVCFeatureEncoder: 13 channels → 16 processed features  
# ✅ Enhanced Enhancer: RGB(3) + metadata(6) + VVC(16) = 25 channels
# ✅ Backward compatibility: works without VVC features
# ✅ Neural architecture: preserves original RGB output
# 🎉 ALL TESTS PASSED!
```

## 🎯 Benefits

1. **Codec Intelligence**: Network understands VVC encoding decisions
2. **Block-aware Processing**: Knows about CTU/CU boundaries and sizes
3. **Quantization Awareness**: Understands compression artifacts
4. **Temporal Patterns**: Access to inter-frame prediction information
5. **Backward Compatible**: Existing models continue to work
6. **Automatic Generation**: VVC features generated during decode_data.sh

## 🔧 Troubleshooting

### No VVC Features Loaded
- Check `fused_maps_dir` path in configuration
- Ensure `fused_maps_poc*.npz` files exist
- Verify sequence names match between chunks and maps

### Network Output Shape Errors
- Ensure `output_block.features: 3` in configuration
- Check that `with_mask: true` if using residual learning

### Memory Issues
- VVC features add ~13×4 bytes per pixel
- Consider smaller batch sizes
- Use gradient checkpointing if needed

---

**Your neural network now has VVC intelligence! 🧠⚡**