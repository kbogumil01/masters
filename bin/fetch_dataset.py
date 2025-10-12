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

SELECTED_VIDEOS = [
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


def process_video(target: str) -> None:
    parts = target.split(".")
    parts[-1] = "yuv"
    dest = ".".join(parts)
    
    # First, get total frame count
    probe_cmd = f'ffprobe -v quiet -select_streams v:0 -count_frames -show_entries stream=nb_frames -csv=p=0 "{target}"'
    
    try:
        result = subprocess.run(probe_cmd, shell=True, capture_output=True, text=True)
        total_frames = int(result.stdout.strip())
        print(f"Total frames in {target}: {total_frames}")
        
        # Calculate middle 64 frames
        if total_frames <= 64:
            # If video has 64 or fewer frames, take all frames
            start_frame = 0
            frames_to_take = total_frames
            print(f"Video too short, taking all {frames_to_take} frames")
        else:
            # Take middle 64 frames
            start_frame = (total_frames - 64) // 2
            frames_to_take = 64
            print(f"Taking 64 frames starting from frame {start_frame} (middle section)")
        
        # FFmpeg command with skip and limit
        ffmpeg_cmd = (
            f"ffmpeg -y -i {target} -vf \"select='between(n,{start_frame},{start_frame + frames_to_take - 1})'\" "
            f"-c:v rawvideo -pixel_format yuv420p -frames:v {frames_to_take} {dest}"
        )
        
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error getting frame count for {target}: {e}")
        print("Falling back to first 64 frames")
        # Fallback to original method
        ffmpeg_cmd = (
            f"ffmpeg -y -i {target} -c:v rawvideo -pixel_format yuv420p -frames:v 64 {dest}"
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
