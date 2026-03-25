import os
from torch import distributed, tensor



def all_reduce_sum(val, device):
    '''  
    given a scalar, sum it across all ranks 
    '''
    total = tensor(val, device=device) 
    distributed.all_reduce(total) # does sum reduction by default...
    return total  

def all_reduce_mean_and_reweight(mean, weight, device):
    '''  
    given a mean and count used to get it, get mean across all items in all ranks

    multiplies mean by weight to get total per rank,
    all_reduces totals and all_reduces weights to get global total and weight,
    then gets quotient to find mean across items 
    '''
    total = tensor(mean, device=device) * weight
    distributed.all_reduce(total) # does sum reduction by default...
    distributed.barrier()
    total_possible = tensor(weight, device=device)
    distributed.all_reduce(total_possible) # does sum reduction by default...
    return total / total_possible  # ...divide by sum of weights to get mean 


class CustomGreweGPUEnv:
    """
    An environment for running on clusters managed by the LSF resource manager.
    """

    def __init__(self):
        os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])


        self.num_processes_per_node = self.world_size
      


