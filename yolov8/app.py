from flask import Flask, request, jsonify
import time
import torch
from ultralytics import YOLO
import os
from PIL import Image
import io
import json

app = Flask(__name__)

model = YOLO("/root/yolov8/yolov8n.pt")

@app.route('/detect', methods=['POST'])
def detect():
    client_server = float(request.headers.get('clientside')) if 'clientside' in request.headers else 'Sender time not found'
    start_time = time.time()
    # calculating client to server propagation time
    client_side_prop = start_time - client_server 
    print(f'Client prop: {client_side_prop}')
    
    if 'img' not in request.files:
        return 'File not received!'

    image = request.files['img']
    if image.filename == '':
        return 'File name not detected!'
    try:
        im_pil = Image.open(image)
        print("Opened the image")
        results = model(im_pil)
        print(f"printing results: {results}")
        nparr = results.render()[0]
        nplist = nparr.tolist()
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"inference time: {processing_time}")
        # embedding current time for client to extract
        response = {
            'image': nplist,
            'proc_time': processing_time,
            'clientsideprop': client_side_prop,
            'serverclientprop': end_time
        }
        print(response)

        #response_json = json.dumps(response)

        return jsonify(response), 200
    
    except Exception as e:
        print(f"Error during detection: {e}")
        return f'Error processing the image: {e}', 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)