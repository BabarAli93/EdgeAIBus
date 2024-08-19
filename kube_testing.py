from kubeframework.utils.kube_utils.kube_cluster import KubeCluster
from kubeframework.utils.kube_utils.descriptors import KubeNode
from kubeframework.utils.kube_utils.utils import (
    construct_pod, construct_service, construct_deployment, 
    generate_random_service_name, get_node_name)
import time
import numpy as np
import csv
import os
import pathlib

fieldnames = ['time', 'cpu_request', 'cpu_usage', 'memory_request', 'memory_usage', 'model']

def generate_csv(path: str, fieldnames: list):
    """ 
    
    This function generates CSV file to log content if does not exist

    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = 'dataset_pod_1s.csv'
    f"{base_name[:-4]}_{timestamp}.csv"

    csv_name = f"{base_name[:-4]}_{timestamp}.csv"
    csv_path = os.path.join(path, csv_name)

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    print('CSV created...')
    return csv_path

path = pathlib.Path(__file__).parent.resolve()
dir_path = os.path.join(path, 'yolov8/logs/small')
csv_path = generate_csv(dir_path, fieldnames)

#######################################################

CONFIG_FILE = '~/.kube/config'

cluster_context = 'gke_essential-sum-426322-v9_europe-west1-b_cluster-1'

cluster = KubeCluster(config_file=CONFIG_FILE, context=cluster_context)

nodes = []
cluster_nodes =  cluster.monitor.get_nodes()

for node in cluster_nodes:
    nodes.append(node)


images = ['babarkhan93/yolo8s:0.1']

""" This function creates a pod and its service in given clusters's
 give node and expost pod with LoadBalancer at port 5000
"""


for i in range(1):
    service_name = generate_random_service_name(service_id=i, node_id=i)
    service_body = construct_service(name=service_name, namespace='prokube', port=5000, targetPort=5000)
    pod_body = construct_pod(name=service_name, image=images[0], request_cpu='1000m',
                           request_mem='1000Mi', limit_cpu='1000m', limit_mem='1000Mi')
    cluster.action.create_pods([pod_body])
    # deployment = construct_deployment(name=service_name, image=images[0], request_cpu='500m',
    #                         request_mem='1000Mi', limit_cpu='500m', limit_mem='1000Mi')
    #cluster.action.create_deployment(deployment=deployment)
    _, ip = cluster.action.create_service(service_body)
    print(ip)


#############################################################################

def pod_stats(pod_name):

    cpu_usage, memory_usage = cluster.monitor.get_pod_metrics_top(pod_name=pod_name)
    cpu_request, memory_request = cluster.monitor.get_pod_resource_requests(pod_name)

    return {
        'time': time.time(),
        'cpu_request':cpu_request, 
        'cpu_usage': cpu_usage, 
        'memory_request':memory_request, 
        'memory_usage':memory_usage,
        'model': 'small'
        }


while True:
    result  = pod_stats(service_name)
    print(result['cpu_usage'])
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(result)
        time.sleep(3)


# print(f"Pod Metrics Testing")
# print(cluster.monitor.get_pods_metrics())
# #
# # Get Nodes metrics
# node_name = get_node_name(nodes[0])
# print(f"Printing Nodes Metrics")
# print(cluster.monitor.get_node_metrics_top(node_name))

# print("VPA Recommendations!")
# #print(cluster.monitor.get_vpa_recommendation())

