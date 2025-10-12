import os
import re
import numpy as np
import cv2
import glob
from tqdm import tqdm
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Metadata:
    file: str
    width: int
    height: int
    frames: int
    profile: str
    qp: int
    alf: bool
    sao: bool
    db: bool


@dataclass
class Frame:
    metadata: Metadata
    frame: int
    is_intra: bool


class Splitter:
    """
    Splits videos into chunks for easy processing
    """

    INFO_HEIGHT_REGEX = re.compile(r"^\s*Height\s*:\s*(\d+)\s*$")
    INFO_WIDTH_REGEX = re.compile(r"^\s*Width\s*:\s*(\d+)\s*$")
    INFO_FRAMES_REGEX = re.compile(r"^\s*Frame count\s*:\s*(\d+)\s*$")
    ENCODED_REGEX = re.compile(
        "^(?P<name>[\d\s\w]+)_(?P<profile>AI|RA)_QP(?P<qp>\d{2})_ALF(?P<alf>\d{1})_DB(?P<db>\d{1})_SAO(?P<sao>\d{1}).yuv"
    )

    METADATA_FORMAT: str = "{name}.*.info"
    DECODED_FORMAT: str = "{file}_{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}.yuv"
    DECODED_LOG_FORMAT: str = (
        "{file}_{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}.yuv.log"
    )
    ORIGINAL_FORMAT: str = "{file}.yuv"
    FILE_FORMAT: str = "yuv"

    FRAME_NAME = "{file}__{height}__{width}/{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}/{frame}_{is_intra}.png"
    ORIG_FRAME_NAME = "{file}/{frame}.png"

    def __init__(
        self,
        data_path: str,
        encoded_path: str,
        frame_folder: str,
        orig_frame_folder: str,
        done_cache: str,
    ) -> None:
        super().__init__()

        self.data_path = data_path
        self.encoded_path = encoded_path

        self.frame_folder = frame_folder
        self.orig_frame_folder = orig_frame_folder
        self.done_cache = done_cache

    def load_intra_frames(self, metadata: Metadata, dirname: str) -> set[int]:
        # Look for decode.log in the directory
        file_path = os.path.join(self.encoded_path, dirname, "decode.log")

        with open(file_path) as f:
            lines = f.read().splitlines()

        lines = [line for line in lines if line.startswith("POC")]
        return {i for i, line in enumerate(lines) if "I-SLICE" in line}

    def split_frames(self) -> None:
        """
        splits frames :)
        """
        dirs = [d for d in os.listdir(self.encoded_path) 
                if os.path.isdir(os.path.join(self.encoded_path, d)) and not d.startswith('.')]
        dirs = sorted(dirs)

        try:
            with open(self.done_cache) as f:
                done = f.read().splitlines()
        except:
            done = []

        for dirname in tqdm(dirs):
            if dirname in done:
                continue

            # Skip RA sequences for now (focus on AI)
            if "_RA_" in dirname:
                continue

            # Check if recon.yuv exists
            recon_path = os.path.join(self.encoded_path, dirname, "recon.yuv")
            if not os.path.exists(recon_path):
                continue

            metadata = self.load_metadata_for(dirname)
            intra_frames = self.load_intra_frames(metadata, dirname)

            video_frames = []

            for frame_num in range(metadata.frames):
                frame = Frame(
                    metadata=metadata,
                    frame=frame_num,
                    is_intra=frame_num in intra_frames or metadata.profile == "AI",
                )
                video_frames.append(frame)

            self.save_frames(video_frames, dirname)

            with open(self.done_cache, "a") as f:
                f.write(f"\n{dirname}")

            print(f"DONE {dirname}")

    def load_metadata_for(self, dirname: str) -> Metadata:
        """
        Loads metadata for given directory
        """
        # Parse directory name: deadline_cif_AI_QP37_ALF1_DB0_SAO1
        m = re.match(self.ENCODED_REGEX, dirname + ".yuv")  # Add .yuv for regex match
        assert m, f"Invalid directory name: {dirname} not matching regex pattern"
        match_group = m.groupdict()

        height = width = frames = None

        fname = glob.glob(
            os.path.join(self.data_path, self.METADATA_FORMAT.format_map(match_group))
        )[0]

        with open(fname) as f:
            for line in f.readlines():
                h = re.match(self.INFO_HEIGHT_REGEX, line)
                height = h.groups()[0] if h else height
                w = re.match(self.INFO_WIDTH_REGEX, line)
                width = w.groups()[0] if w else width
                fr = re.match(self.INFO_FRAMES_REGEX, line)
                frames = fr.groups()[0] if fr else frames

        assert isinstance(height, str)
        assert isinstance(width, str)
        print(match_group["alf"])

        return Metadata(
            file=match_group["name"],
            width=int(width),
            height=int(height),
            frames=64,
            profile=match_group["profile"],
            qp=int(match_group["qp"]),
            alf=bool(int(match_group["alf"])),
            db=bool(int(match_group["db"])),
            sao=bool(int(match_group["sao"])),
        )

    def save_frames(self, frames: list[Frame], dirname: str) -> None:
        """
        Splits chunks
        """
        metadata = frames[0].metadata
        nh = metadata.height * 3 // 2

        orig_file_path = os.path.join(
            self.data_path, self.ORIGINAL_FORMAT.format_map(asdict(metadata))
        )
        # Point to recon.yuv in the directory
        file_path = os.path.join(self.encoded_path, dirname, "recon.yuv")

        with open(file_path, "rb") as f:
            buff = np.frombuffer(f.read(), dtype=np.uint16)

        with open(orig_file_path, "rb") as f:
            orig_buff = np.frombuffer(f.read(), dtype=np.uint8)

        buff = np.round(buff / 4).astype(np.uint8)
        buff = np.resize(buff, (metadata.frames, nh * metadata.width))

        orig_buff = np.resize(orig_buff, (metadata.frames, nh * metadata.width))

        for frame_num in tqdm(range(metadata.frames)):
            frame = buff[frame_num]
            frame = self.upsample_uv(frame, metadata.width, metadata.height)

            orig_frame = orig_buff[frame_num]
            orig_frame = self.upsample_uv(orig_frame, metadata.width, metadata.height)

            frame_data = [f for f in frames if f.frame == frame_num][0]

            frame_name = self.FRAME_NAME.format_map(
                dict(**asdict(metadata), **asdict(frame_data))
            )
            orig_frame_name = self.ORIG_FRAME_NAME.format_map(
                dict(**asdict(metadata), **asdict(frame_data))
            )

            fname = os.path.join(self.frame_folder, frame_name)
            folder = os.path.dirname(fname)
            Path(folder).mkdir(parents=True, exist_ok=True)

            if not os.path.exists(fname):
                with open(fname, "wb") as fo:
                    fo.write(frame.tobytes())

            fname = os.path.join(self.orig_frame_folder, orig_frame_name)
            folder = os.path.dirname(fname)
            Path(folder).mkdir(parents=True, exist_ok=True)

            if not os.path.exists(fname):
                with open(fname, "wb") as fo:
                    fo.write(orig_frame.tobytes())

    def upsample_uv(self, frame_buffer, width, height):
        i = width * height
        Y = frame_buffer[0:i]
        Y = np.reshape(Y, (height, width))

        uv_size = width * height // 4
        V = frame_buffer[i : i + uv_size]
        V = np.reshape(V, (height // 2, width // 2))
        V = cv2.resize(V, (width, height))

        i += uv_size
        U = frame_buffer[i:]
        U = np.reshape(U, (height // 2, width // 2))
        U = cv2.resize(U, (width, height))

        return np.dstack([Y, U, V])


if __name__ == "__main__":
    import sys

    s = Splitter(*sys.argv[1:])
    s.split_frames()