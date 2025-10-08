import cv2
import argparse
from pathlib import Path

def preprocess_ronchi(input_path: Path, size: int = 320):
    """Preprocess a Ronchi image for model inference.

    Steps:
    1. Read as grayscale.
    2. Apply histogram equalization to normalize contrast.
    3. Resize to match model training size.
    """
    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    # Equalize brightness and contrast
    img = cv2.equalizeHist(img)

    # Resize to training resolution
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    # Save next to original
    output_path = input_path.with_name(f"{input_path.stem}_preprocessed.png")
    cv2.imwrite(str(output_path), img)
    print(f"Saved preprocessed image to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Ronchi image for model inference.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--size", type=int, default=320, help="Resize dimension (default=320)")
    args = parser.parse_args()

    preprocess_ronchi(Path(args.image_path), size=args.size)
