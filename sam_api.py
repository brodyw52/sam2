from flask import Flask, request, send_file
from segment_anything import SamPredictor, sam_model_registry
import torch
import cv2
import numpy as np

app = Flask(__name__)
model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to("cpu")
predictor = SamPredictor(model)

@app.route("/segment", methods=["POST"])
def segment():
    image_file = request.files["image"]
    image_path = "input.jpg"
    mask_path = "mask.png"
    image_file.save(image_path)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=np.array([[100, 100]]), 
        point_labels=np.array([1]), 
        multimask_output=False
    )
    mask = masks[0].astype(np.uint8) * 255
    cv2.imwrite(mask_path, mask)
    return send_file(mask_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
