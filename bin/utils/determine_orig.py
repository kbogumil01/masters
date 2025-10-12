import os


ORIG_FOLDER = "test_data"


def determine_orig(f: str) -> str:
    result = os.path.join(ORIG_FOLDER, f"{f}.yuv")

    assert os.path.exists(result)
    return result