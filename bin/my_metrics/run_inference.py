#!/usr/bin/env python3
"""
Step 1: Run inference on test frames and save enhanced frames as .pt files.

Usage:
    python -m bin.my_metrics.run_inference \\
        --checkpoint checkpoints/epoch=787.ckpt \\
        --config experiments/enhancer/baseline_0501.yaml \\
        --input-dir videos_test/test_frames_REAL \\
        --output-dir videos_test/enhanced_pt
"""

import torch
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from enhancer.config import Config
from enhancer.models.enhancer import Enhancer


def load_model(checkpoint_path: str, config_path: str, device: str = 'cuda'):
    """Load enhancer model from checkpoint."""
    config = Config.load(config_path)
    enhancer = Enhancer(config=config.enhancer)
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
        # Remove 'enhancer.' prefix if present
        state = {k.replace("enhancer.", ""): v for k, v in state.items() if k.startswith("enhancer.")}
        enhancer.load_state_dict(state, strict=True)
    else:
        enhancer.load_state_dict(ckpt, strict=True)
    
    enhancer.to(device)
    enhancer.eval()
    return enhancer


def build_metadata_tensor(seq_meta: dict, is_intra: bool, device: str) -> torch.Tensor:
    """Build metadata tensor from seq_meta dict."""
    profile = str(seq_meta.get("profile", "AI")).upper()
    profile_ai = 1.0 if "AI" in profile else 0.0
    qp = float(seq_meta.get("qp", 32))
    qp_n = qp / 64.0
    alf = float(seq_meta.get("alf", 0))
    sao = float(seq_meta.get("sao", 0))
    db = float(seq_meta.get("db", 0))
    is_intra_f = float(is_intra)
    
    meta = torch.tensor([profile_ai, qp_n, alf, sao, db, is_intra_f], dtype=torch.float32)
    return meta.view(1, 6, 1, 1).to(device)


def process_sequence(enhancer, seq_dir: Path, output_dir: Path, device: str):
    """Process all frames in a sequence directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_files = sorted(seq_dir.glob("frame_poc*.pt"))
    
    for frame_file in tqdm(frame_files, desc=f"  {seq_dir.name}", leave=False):
        # Load frame
        data = torch.load(frame_file, map_location="cpu")
        
        # Get tensors
        decoded = data["chunk"].float() / 255.0
        decoded = decoded.unsqueeze(0).to(device)
        
        seq_meta = data.get("seq_meta", {})
        is_intra = data.get("is_intra", 1)
        metadata = build_metadata_tensor(seq_meta, is_intra, device)
        
        # VVC features if available
        vvc_feat = None
        if "vvc_features" in data and data["vvc_features"] is not None:
            vvc_feat = data["vvc_features"].float().unsqueeze(0).to(device)
        
        # Enhance
        with torch.no_grad():
            enhanced = enhancer(decoded, metadata, vvc_feat)
        
        # Save enhanced frame (as float32 0-1 to match Piotr's format)
        output_data = {
            "seq_meta": seq_meta,
            "poc": data["poc"],
            "chunk": enhanced.squeeze(0).cpu(),  # [3, H, W] float32 0-1
            "is_intra": is_intra,
        }
        
        output_file = output_dir / frame_file.name
        torch.save(output_data, output_file)


def main():
    parser = argparse.ArgumentParser(description="Run inference on test frames")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to checkpoint")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--input-dir", required=True, help="Input directory with decoded frames")
    parser.add_argument("--output-dir", required=True, help="Output directory for enhanced frames")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}")
    enhancer = load_model(args.checkpoint, args.config, args.device)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Find all sequence directories
    seq_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    print(f"Processing {len(seq_dirs)} sequences...")
    for seq_dir in tqdm(seq_dirs, desc="Sequences"):
        out_seq_dir = output_dir / seq_dir.name
        process_sequence(enhancer, seq_dir, out_seq_dir, args.device)
    
    print(f"Done! Enhanced frames saved to {output_dir}")


if __name__ == "__main__":
    main()
