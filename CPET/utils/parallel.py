import numpy as np

from CPET.utils.calculator import (
    propagate_topo_dev,
    propagate_topo_dev_batch, 
    Inside_Box,
    compute_curv_and_dist
)

def task_base(x_0, n_iter, x, Q, step_size, dimensions):
    """
    Takes: 
        x_0(array) - (3, 1) array of box position 
        n_iter(int) - number of iterations of propagation for this slip
        x(np array) - positions of charges
        Q(np array) - charge values 
        step_size(float) - step size of each step 
        dimensions(array) - box limits
    """

    #print("start task")
    x_init = x_0
    for j in range(n_iter):
        #if j == 0: 
        #    debug=True
        #else: 
        #    debug = False
        # x_0 = propagate_topo_dev(x_0, self.x, self.Q, self.step_size)

        x_0 = propagate_topo_dev(x_0, x, Q, step_size)
        if not Inside_Box(x_0, dimensions):
            # count += 1
            break

    x_init_plus = propagate_topo_dev(x_init, x, Q, step_size)
    x_init_plus_plus = propagate_topo_dev(x_init_plus, x, Q, step_size)
    x_0_plus = propagate_topo_dev(x_0, x, Q, step_size)
    x_0_plus_plus = propagate_topo_dev(x_0_plus, x, Q, step_size)

    result = compute_curv_and_dist(
        x_init, x_init_plus, x_init_plus_plus, x_0, x_0_plus, x_0_plus_plus
    )
    return result


def task(x_0, n_iter, x, Q, step_size, dimensions):
    """
    Takes: 
        x_0(array) - (3, 1) array of box position 
        n_iter(int) - number of iterations of propagation for this slip
        x(np array) - positions of charges
        Q(np array) - charge values 
        step_size(float) - step size of each step 
        dimensions(array) - box limits
    """

    #print("start task")
    x_init = x_0
    for j in range(n_iter):
        #if j == 0: 
        #    debug=True
        #else: 
        #    debug = False
        # x_0 = propagate_topo_dev(x_0, self.x, self.Q, self.step_size)

        x_0 = propagate_topo_dev(x_0, x, Q, step_size)
        if not Inside_Box(x_0, dimensions):
            # count += 1
            break

    x_init_plus = propagate_topo_dev(x_init, x, Q, step_size)
    x_init_plus_plus = propagate_topo_dev(x_init_plus, x, Q, step_size)
    x_0_plus = propagate_topo_dev(x_0, x, Q, step_size)
    x_0_plus_plus = propagate_topo_dev(x_0_plus, x, Q, step_size)

    result = compute_curv_and_dist(
        x_init, x_init_plus, x_init_plus_plus, x_0, x_0_plus, x_0_plus_plus
    )
    return result


def task_batch(x_0_list, n_iter, x, Q, step_size, dimensions):
    x_init = x_0_list
    mask_list = None
    n_iter_max = np.max(n_iter)
    for j in range(n_iter_max):
        #print("mask list {}".format(mask_list))
        # x_0 = propagate_topo_dev(x_0, self.x, self.Q, self.step_size)
        
        x_0_list = propagate_topo_dev_batch(x_0_list, x, Q, step_size, mask_list=mask_list)
        #print("passed first iter")
        inside_box_list = [bool(Inside_Box(x, dimensions)) for x in x_0_list]
        # filter list consists of indicies where n_iter[i] < j
        rand_stop_cond = [bool(j <= n_iter[i]) for i in range(len(n_iter))]

        #print("inside box: {}".format(inside_box_list))
        #print("rand stop: {}".format(rand_stop_cond))
        # create mask list where it's true is inside_box or filter_list
        mask_list_contra = [bool(rand_stop_cond[i] and inside_box_list[i]) for i in range(len(n_iter))]
        mask_list = [not temp for temp in mask_list_contra]
        
        #print("mask list: ", mask_list)
        #print("mask list contra: ", mask_list_contra)
        if all(mask_list):
            #print("breaking")
            break
            
        
    
    x_init_plus = propagate_topo_dev_batch(x_init, x, Q, step_size)
    x_init_plus_plus = propagate_topo_dev_batch(x_init_plus, x, Q, step_size)
    x_0_plus = propagate_topo_dev_batch(x_0_list, x, Q, step_size)
    x_0_plus_plus = propagate_topo_dev_batch(x_0_plus, x, Q, step_size)

    #result_list = compute_curv_and_dist(
    #    x_init, x_init_plus, x_init_plus_plus, x_0, x_0_plus, x_0_plus_plus
    #)

    result_list = []
    # not currently batched
    for ind in range(len(x_0_plus)): 
        result = compute_curv_and_dist(
            x_init[ind], x_init_plus[ind], x_init_plus_plus[ind], x_0_list[ind], x_0_plus[ind], x_0_plus_plus[ind]
        )
        result_list.append(result)

    return result_list