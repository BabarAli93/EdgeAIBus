import requests
import pathlib
import os
import json
import numpy as np
from PIL import Image
import time
import csv
from tenacity import retry, wait_exponential, stop_after_attempt

class WorkloadGenerator:

    def __init__(self):

        path = pathlib.Path(__file__).parent.parent.resolve()
        self.dir_path = path.joinpath('images')
        self.save_path = path.joinpath('results')
        self.logs_path = path.joinpath('logs')
        self.image_path = self.dir_path.joinpath('6.jpg')

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
    def client_request(self, ip):
        item = '6.jpg'
        s_time = time.time()
        headers = {'clientside': str(s_time)}
        with open(self.image_path, 'rb') as f:
            files = {'img': f}
            response = requests.post(f'http://{ip}:5000/detect', files=files, headers=headers, timeout=(10,30))
            
        if response.status_code == 200:
            json_response = json.loads(response.text)
            img_list = json_response['detection_results']
            img_np = np.array(img_list, dtype=np.uint8)
            image = Image.fromarray(img_np)
            image.save(os.path.join(self.save_path, f"{item}.jpeg"))
            
            return {
                'time': time.time(),
                'processing_delay(s)': json_response['proc_time'],
            }
        else:
            print('Unsuccessful response!')
            return None 