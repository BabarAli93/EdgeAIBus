import numpy as np
from typing import List

class Mapper:

    def __init__(self, hosts:int, unique_cores:List[int],  
                 model_versions: int = 2):
        self.num_hosts = hosts
        self.model_versions = model_versions
        self.mapper_range = self.num_hosts * self.model_versions   # given 6 hosts and 2 models, we will have total 12 values
        self.container_unique_cpu = unique_cores

    def _int_to_tuple_mapper(self, n):
        binary_value = n % 2
        first_value = (n // 2) % (self.num_hosts + 1)
        
        return (first_value, binary_value)

    def _generate_mapping_dict(self):
        """ index will be the action coming from step and value is 
        its corresponding (host_id, model_version)"""
        int_list = range(self.mapper_range)  # Create a list of integers from num_hosts * num_version (0-11 in curent case)
        tuple_list = [self._int_to_tuple_mapper(n) for n in int_list]
        return np.array(tuple_list)

    def get_tuple(self, n):
        return self.mapping_dict.get(n)
    
    def _get_core_model_mapper(self):

        """
        this mapper will get us the index of numpy array from self.yolodata
        Resultant Numpy array: values will be the tuple(core, model) and index will be the index of self.yolodata correponding core, model array
       array([[ 500,    0],
       [ 500,    1],
       [1000,    0],
       [1000,    1],
       [1500,    0],
       [1500,    1]])

        """
        repeated_arr = np.repeat(self.container_unique_cpu, self.model_versions)
        binary_arr = np.array([0, 1] * len(self.container_unique_cpu))
        result = np.column_stack((repeated_arr, binary_arr))

        return result
