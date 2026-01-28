#!/usr/bin/env python3
"""
Utility functions for metrics pipeline.
"""
import re
import os
from pathlib import Path


# Known video resolutions
KNOWN_RESOLUTIONS = {
    "Johnny_1280x720_60": (1280, 720),
    "bridge_far_cif": (352, 288),
    "stefan_cif": (352, 288),
    "students_qcif": (176, 144),
    "tractor_1080p25": (1920, 1080),
}


def parse_sequence_name(dirname: str) -> dict:
    """
    Parse sequence directory name like:
    Johnny_1280x720_60_AI_QP32_ALF0_DB0_SAO0
    
    Returns dict with: name, profile, qp, alf, db, sao
    """
    # Pattern: name_PROFILE_QPxx_ALFx_DBx_SAOx
    match = re.match(r'(.+?)_(AI|RA)_QP(\d+)_ALF(\d)_DB(\d)_SAO(\d)', dirname)
    
    if not match:
        return None
    
    return {
        'name': match.group(1),
        'profile': match.group(2),
        'qp': int(match.group(3)),
        'alf': int(match.group(4)),
        'db': int(match.group(5)),
        'sao': int(match.group(6)),
    }


def get_resolution(seq_name: str) -> tuple:
    """Get (width, height) for a sequence name."""
    if seq_name in KNOWN_RESOLUTIONS:
        return KNOWN_RESOLUTIONS[seq_name]
    
    # Try to parse from name like "Johnny_1280x720_60"
    match = re.search(r'(\d+)x(\d+)', seq_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    
    # CIF/QCIF patterns
    if 'qcif' in seq_name.lower():
        return (176, 144)
    if 'cif' in seq_name.lower():
        return (352, 288)
    if '1080p' in seq_name.lower():
        return (1920, 1080)
    if '720p' in seq_name.lower():
        return (1280, 720)
    
    raise ValueError(f"Unknown resolution for sequence: {seq_name}")


def get_original_yuv_path(seq_name: str, orig_folder: str = "videos_test/test_dataset") -> str:
    """Get path to original YUV file."""
    return os.path.join(orig_folder, f"{seq_name}.yuv")


def get_bitrate_from_encoder_log(log_path: str) -> float:
    """
    Parse bitrate from vvenc encoder log.
    Format: vvenc [info]: ... bitrate=XXXX.XX kbps
    """
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # vvenc format
    match = re.search(r'bitrate[=:\s]+(\d+\.?\d*)\s*kbps', content, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    return None


def count_frames_in_sequence(seq_dir: str) -> int:
    """Count .pt frame files in a sequence directory."""
    return len(list(Path(seq_dir).glob("frame_poc*.pt")))
