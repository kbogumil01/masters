#! /usr/bin/python3

# This script fetches selected data from YT-UGC dataset and saves it
# to data folder.

from tqdm import tqdm
import os
import requests
import argparse
import subprocess


CHUNK_SIZE = 128 * 1024  # 128 KB
DEFAULT_FOLDER = "data"
BASE_URL = "https://media.xiph.org/video/derf/y4m/"

SELECTED_VIDEOS = [ # jedno z tych usunąć bo nie pasuje!
    "720p50_mobcal_ter.y4m",
    "720p50_parkrun_ter.y4m",
    "720p50_shields_ter.y4m",
    "720p5994_stockholm_ter.y4m",
    "FourPeople_1280x720_60.y4m",
    "KristenAndSara_1280x720_60.y4m",
    "akiyo_cif.y4m",
    "blue_sky_1080p25.y4m",
    "bowing_qcif.y4m",
    "bridge_close_cif.y4m",
    "bus_cif.y4m",
    "carphone_qcif.y4m",
    "city_cif.y4m",
    "claire_qcif.y4m",
    "coastguard_cif.y4m",
    "container_cif.y4m",
    "crew_cif.y4m",
    "deadline_cif.y4m",
    "ducks_take_off_420_720p50.y4m",
    "flower_cif.y4m",
    "football_sif.y4m",
    "foreman_cif.y4m",
    "garden_sif.y4m",
    "grandma_qcif.y4m",
    "hall_monitor_cif.y4m",
    "harbour_4cif.y4m",
    "highway_qcif.y4m",
    "husky_cif.y4m",
    "ice_4cif.y4m",
    "in_to_tree_420_720p50.y4m",
    "mad900_cif.y4m",
    "miss_am_qcif.y4m",
    "mobile_cif.y4m",
    "mother_daughter_cif.y4m",
    "mthr_dotr_qcif.y4m",
    "news_cif.y4m",
    "old_town_cross_420_720p50.y4m",
    "pamphlet_cif.y4m",
    "paris_cif.y4m",
    "park_joy_420_720p50.y4m",
    "riverbed_1080p25.y4m",
    "salesman_qcif.y4m",
    "sign_irene_cif.y4m",
    "silent_cif.y4m",
    "sintel_trailer_2k_480p24.y4m",
    "soccer_4cif.y4m",
    "station2_1080p25.y4m",
    "students_cif.y4m",
    "suzie_qcif.y4m",
    "tempete_cif.y4m",
    "tennis_sif.y4m",
    "trevor_qcif.y4m",
    "tt_sif.y4m",
    "waterfall_cif.y4m",
]


def get_y4m_frame_count(filename: str) -> int:
    """Get frame count from Y4M file by counting FRAME headers"""
    try:
        frame_count = 0
        with open(filename, 'rb') as f:
            # Read header line to get frame dimensions
            header = f.readline().decode('ascii', errors='ignore')
            if not header.startswith('YUV4MPEG2'):
                return 0
            
            # Parse width and height from header
            width = height = 0
            for param in header.split():
                if param.startswith('W'):
                    width = int(param[1:])
                elif param.startswith('H'):
                    height = int(param[1:])
            
            if width == 0 or height == 0:
                return 0
            
            # Calculate frame size (YUV420p format: Y + U/4 + V/4)
            frame_size = width * height * 3 // 2
            
            # Count FRAME headers
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith(b'FRAME'):
                    frame_count += 1
                    # Skip frame data
                    f.seek(frame_size, 1)  # Skip relative to current position
                    
        return frame_count
    except Exception as e:
        print(f"Error reading Y4M file: {e}")
        return 0


def process_video(target: str) -> None:
    parts = target.split(".")
    parts[-1] = "yuv"
    dest = ".".join(parts)
    
    # First, get total frame count using Y4M parsing (faster than ffprobe)
    total_frames = get_y4m_frame_count(target)
    
    if total_frames == 0:
        print(f"Could not determine frame count for {target}, trying ffprobe...")
        # Fallback to ffprobe without -count_frames (faster)
        probe_cmd = f'ffprobe -v quiet -select_streams v:0 -show_entries stream=nb_frames -csv=p=0 "{target}"'
        try:
            result = subprocess.run(probe_cmd, shell=True, capture_output=True, text=True)
            if result.stdout.strip():
                total_frames = int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            pass
    
    if total_frames > 0:
        print(f"Total frames in {target}: {total_frames}")
        
        # Calculate frames to take (skip first 64, then take next 64)
        if total_frames <= 64:
            # If video has 64 or fewer frames, take all frames
            start_frame = 0
            frames_to_take = total_frames
            print(f"Video too short, taking all {frames_to_take} frames")
        elif total_frames <= 128:
            # If video has between 65-128 frames, take from frame 32 onwards
            start_frame = min(32, total_frames - 64)
            frames_to_take = min(64, total_frames - start_frame)
            print(f"Medium length video, taking {frames_to_take} frames starting from frame {start_frame}")
        else:
            # Take frames 64-127 (skip first 64, then take next 64)
            start_frame = 64
            frames_to_take = 64
            print(f"Taking frames 64-127 (skipping first 64 frames)")
        
        # FFmpeg command with skip and limit
        ffmpeg_cmd = (
            f"ffmpeg -y -i {target} -vf \"select='between(n,{start_frame},{start_frame + frames_to_take - 1})'\" "
            f"-c:v rawvideo -pixel_format yuv420p -frames:v {frames_to_take} {dest}"
        )
    else:
        print(f"Could not determine frame count for {target}, using frames 64-127")
        # Fallback: skip first 64 frames, take next 64
        ffmpeg_cmd = (
            f"ffmpeg -y -i {target} -vf \"select='between(n,64,127)'\" "
            f"-c:v rawvideo -pixel_format yuv420p -frames:v 64 {dest}"
        )
    
    mediainfo_cmd = f"mediainfo -f {target} > {target}.info"
    subprocess.call(mediainfo_cmd, shell=True)
    subprocess.call(ffmpeg_cmd, shell=True)
    os.remove(target)


def download_videos(target: str) -> None:
    for video in tqdm(SELECTED_VIDEOS):
        url = os.path.join(BASE_URL, video)
        print(f"downloading {video}...")
        target_filename = os.path.join(target, video)

        if os.path.exists(target_filename):
            should_proceed = (
                input(f"file {target_filename} already exists, overwrite it? (y/n)")
                .strip()
                .upper()
            )
            if should_proceed != "Y":
                continue

        with requests.get(url, stream=True) as data:
            data.raise_for_status()

            with open(target_filename, "wb") as f:
                for chunk in tqdm(
                    data.iter_content(chunk_size=CHUNK_SIZE), leave=False
                ):
                    f.write(chunk)

        process_video(target_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", action="store", default=DEFAULT_FOLDER)

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    download_videos(args.dir)
