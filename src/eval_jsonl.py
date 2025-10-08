import argparse, json, csv, math
from PIL import Image
import torch
import torchvision.transforms as T

from src.model import RonchiNet

def load_model(weights_path: str, image_size: int = 320, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(weights_path, map_location=device)
    model = RonchiNet()
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    tfm = T.Compose([T.Grayscale(1), T.Resize((image_size, image_size)), T.ToTensor()])
    return model, device, tfm

def infer(model, device, tfm, image_path):
    im = Image.open(image_path).convert("L")
    x = tfm(im).unsqueeze(0).to(device)
    with torch.no_grad():
        yhat = model(x).cpu().numpy().flatten()[0]
    return float(yhat)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--image_size", type=int, default=320)
    ap.add_argument("--out", default="eval_predictions.csv")
    args = ap.parse_args()

    model, device, tfm = load_model(args.weights, args.image_size)

    y_true, y_pred, rows = [], [], []
    with open(args.manifest) as f:
        for line in f:
            row = json.loads(line)
            path = row["image"]
            meta = row.get("meta", {})
            labels = row.get("labels", {})
            if "p_corr" not in labels:
                continue
            y = float(labels["p_corr"])
            yhat = infer(model, device, tfm, path)
            y_true.append(y)
            y_pred.append(yhat)
            rows.append({
                "image": path, "f": meta.get("f"), "offset": meta.get("offset"), "lpi": meta.get("lpi"),
                "p_corr_true": y, "p_corr_pred": yhat, "abs_err": abs(y - yhat)
            })

    if not y_true:
        print("No labeled rows found.")
        return

    mae = sum(abs(a-b) for a,b in zip(y_true, y_pred)) / len(y_true)
    rmse = math.sqrt(sum((a-b)**2 for a,b in zip(y_true, y_pred)) / len(y_true))
    print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  (n={len(y_true)})")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote per-image results → {args.out}")

if __name__ == "__main__":
    main()
