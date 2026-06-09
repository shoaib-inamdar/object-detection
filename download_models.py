import os
import sys
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

FILES = {
    "yolov4-tiny.weights": {
        "url": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
        "size_mb": 23.1,
        "md5": None,
    },
    "yolov4-tiny.cfg": {
        "url": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        "size_mb": 0.003,
        "md5": None,
    },
    "coco.names": {
        "url": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
        "size_mb": 0.001,
        "md5": None,
    },
}


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  ({mb_down:.1f}/{mb_total:.1f} MB)")
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloaded {mb_down:.1f} MB...")
    sys.stdout.flush()


def download_file(url, dest_path):
    print(f"\n  URL : {url}")
    print(f"  Dest: {dest_path}")
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=_progress_hook)
        print()
        return True
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        return False


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 60)
    print("  YOLOv4-tiny Model Downloader")
    print("=" * 60)
    all_ok = True
    for filename, info in FILES.items():
        dest = os.path.join(MODEL_DIR, filename)
        if os.path.isfile(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"\n[SKIP] {filename} already exists ({size_mb:.2f} MB)")
            continue
        print(f"\n[DOWNLOAD] {filename} (~{info['size_mb']:.1f} MB)")
        success = download_file(info["url"], dest)
        if success and os.path.isfile(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"  [OK] Saved {filename} ({size_mb:.2f} MB)")
        else:
            print(f"  [FAIL] Could not download {filename}")
            all_ok = False
    print("\n" + "=" * 60)
    if all_ok:
        print("  All model files are ready!")
        print("  Run:  python main.py")
    else:
        print("  Some downloads failed. Please retry or download manually.")
    print("=" * 60)


if __name__ == "__main__":
    main()
