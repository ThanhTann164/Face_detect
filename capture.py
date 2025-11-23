import argparse
import os
import time
from typing import Optional

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture face images from webcam")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("Dataset", "FaceData", "raw", "tan"),
        help="Output directory to save captured images",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--max", type=int, default=100, help="Max number of images to capture"
    )
    return parser.parse_args()


def open_camera(device_index: int) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        return None
    return cap


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    capture = open_camera(args.device)
    if capture is None:
        print(f"Cannot open camera device {args.device}. Try another index or check permissions.")
        return

    existing = [f for f in os.listdir(args.out) if f.lower().endswith((".jpg", ".png"))]
    saved_count = len(existing)

    print("Press 'c' to capture, 'q' to quit")
    print(f"Saving to: {os.path.abspath(args.out)}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            cv2.imshow("Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                timestamp_ms = int(time.time() * 1000)
                filename = os.path.join(args.out, f"{timestamp_ms}.jpg")
                if cv2.imwrite(filename, frame):
                    saved_count += 1
                    print(f"Saved: {filename} ({saved_count})")
                else:
                    print("Failed to save image.")

                if saved_count >= args.max:
                    print(f"Reached max images: {args.max}")
                    break

            elif key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


