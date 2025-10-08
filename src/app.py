import argparse, json, torch, gradio as gr
from PIL import Image
import torchvision.transforms as T
from src.model import RonchiNet
from src.topfix import recommend_fix

def load_model(weights_path: str, image_size: int = 320, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(weights_path, map_location=device)
    model = RonchiNet()
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    tfm = T.Compose([T.Grayscale(1), T.Resize((image_size, image_size)), T.ToTensor()])
    return model, device, tfm

def predict(im: Image.Image, f: float, offset: float, lpi: int, model, device, tfm):
    x = tfm(im.convert("L")).unsqueeze(0).to(device)
    with torch.no_grad():
        yhat = model(x).cpu().numpy().flatten()[0]
    action, rationale = recommend_fix(yhat)
    result = {
        "metadata": {"f": f, "offset": offset, "lpi": lpi},
        "prediction": {"p_corr": float(yhat)},
        "recommendation": {"action": action, "rationale": rationale}
    }
    # Return scalar + pretty JSON string
    return float(yhat), json.dumps(result, indent=2)

def main(args):
    model, device, tfm = load_model(args.weights, args.image_size)
    with gr.Blocks() as demo:
        gr.Markdown("# Ronchi Parabolic Correction (POC) — Regression")
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Ronchi image")
                f = gr.Number(value=4.0, precision=2, label="f/# (f)")
                offset = gr.Number(value=-0.25, precision=3, label="Offset (inches; negative=inside ROC)")
                lpi = gr.Number(value=100, precision=0, label="Grating LPI")
                btn = gr.Button("Analyze")
            with gr.Column():
                p_corr = gr.Number(label="Predicted p_corr (1.0 = perfect parabola)")
                # Use Textbox instead of Code/JSON to dodge schema edge-cases
                json_out = gr.Textbox(label="Details / Recommendation", lines=16)

        btn.click(lambda im, fval, off, lp: predict(im, fval, off, lp, model, device, tfm),
                  inputs=[img, f, offset, lpi], outputs=[p_corr, json_out])

    # Bind to loopback to satisfy localhost checks; expose --share if needed
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=True,
        share=args.share,
        show_api=False
    )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--image_size", type=int, default=320)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="127.0.0.1")  # loopback by default
    p.add_argument("--share", action="store_true", help="Create a Gradio share URL")
    main(p.parse_args())
