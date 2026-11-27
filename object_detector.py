import os

from ultralytics import YOLO
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image

MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")

app = Flask(__name__)
# load the YOLO26 model once at startup rather than on every request
model = YOLO(MODEL_PATH)


@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("templates/index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():
    """
        Handler of /detect POST endpoint
        Receives uploaded file with a name "image_file", passes it
        through the YOLO26 object detection network and returns an array
        of bounding boxes.
        :return: a JSON array of objects bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(buf.stream)
    return jsonify(boxes)


def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through the YOLO26 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    results = model.predict(Image.open(buf).convert("RGB"), verbose=False)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = int(box.cls[0].item())
        prob = box.conf[0].item()
        prob_percentage = f"{prob * 100:.2f}%"
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob_percentage
        ])

    return output


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8080)
