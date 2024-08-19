from flask import Flask, request, jsonify
import time
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
from kubernetes import client, config
from kubernetes.client.rest import ApiException

app = Flask(__name__)

model = YOLO("/root/yolov8/yolov8m.pt")

config.load_kube_config()
# Initialize Kubernetes API clients
v1 = client.CoreV1Api()
metrics_api = client.CustomObjectsApi()

def get_pod_resource_usage(namespace, pod_name):
    try:
        # Get pod metrics
        metrics = metrics_api.get_namespaced_custom_object(
            "metrics.k8s.io", "v1beta1", namespace, "pods", pod_name
        )

        cpu_usage = 0
        memory_usage = 0
        for container in metrics['containers']:
            # Convert the CPU usage from nanocores to millicores
            cpu_usage += int(container['usage']['cpu'].rstrip('n')) / 1e6
            # Convert the memory usage from Ki to Mi
            memory_usage += int(container['usage']['memory'].rstrip('Ki')) / 1024
        
        return cpu_usage, memory_usage
    except ApiException as e:
        print(f"Error fetching metrics: {e}")
        return None, None
    
def get_pod_resource_requests(namespace, pod_name):
    try:
        # Get pod spec
        pod = v1.read_namespaced_pod(pod_name, namespace)
        
        cpu_request = 0
        memory_request = 0
        for container in pod.spec.containers:
            if container.resources.requests:
                cpu_request += int(container.resources.requests.get('cpu', '0').rstrip('n'))
                memory_request += int(container.resources.requests.get('memory', '0').rstrip('Ki'))
        
        return cpu_request, memory_request
    except ApiException as e:
        print(f"Error fetching pod spec: {e}")
        return None, None

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
        print("Opened the image")
        
        # Run the YOLO model on the image
        results = model(im_pil)
        print(f"Results: {results}")
        
        # Render the results and convert them to a format suitable for JSON
        annotated_img_np = np.array(results[0].plot())

        end_time = time.time()

        # Get Kubernetes pod resource usage and requests
        namespace = os.getenv('NAMESPACE', 'prokube')  # Use a default namespace if not set
        pod_name = os.getenv('HOSTNAME')  # Kubernetes sets the pod name in the HOSTNAME environment variable
        
        cpu_usage, memory_usage = get_pod_resource_usage(namespace, pod_name)
        cpu_request, memory_request = get_pod_resource_requests(namespace, pod_name)

        response = {
            'detection_results': annotated_img_np.tolist(),
            'clientsideprop': client_side_prop,
            'proc_time': end_time - start_time,
            'serverclientprop': end_time,
            'cpu_usage': cpu_usage,
            'cpu_request': cpu_request,
            'memory_usage': memory_usage,
            'memory_request': memory_request
        }

        return jsonify(response), 200
    
    except Exception as e:
        print(f"Error during detection: {e}")
        return f'Error processing the image: {e}', 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)