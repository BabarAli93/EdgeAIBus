from flask import Flask, request, jsonify
import time
from ultralytics import YOLO
from PIL import Image
import numpy as np


app = Flask(__name__)

model = YOLO("/root/yolov8/yolov8l.pt")

@app.route('/detect', methods=['POST'])
def detect():
    # Retrieve client-side timestamp
    client_server = float(request.headers.get('clientside')) if 'clientside' in request.headers else 'Sender time not found'
    start_time = time.time()
    # calculating client to server propagation time
    client_side_prop = start_time - client_server 
    print(f'Client prop: {client_side_prop}')

    #Check if the file part is present in the request
    if 'img' not in request.files:
        return 'File not received!', 400

    image = request.files['img']
    if image.filename == '':
        return 'File name not detected!', 400
    
    try:
        # Open the image
        im_pil = Image.open(image)        
        # Run the YOLO model on the image
        results = model(im_pil)        
        # Render the results and convert them to a format suitable for JSON
        annotated_img_np = np.array(results[0].plot())

        end_time = time.time()

        response = {
            'detection_results': annotated_img_np.tolist(),
            'clientsideprop': client_side_prop,
            'proc_time': end_time - start_time,
            'serverclientprop': end_time
        }

        return jsonify(response), 200
    
    except Exception as e:
        print(f"Error during detection: {e}")
        return f'Error processing the image: {e}', 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)