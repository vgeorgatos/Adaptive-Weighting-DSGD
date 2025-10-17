import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from collections import OrderedDict


'A code base that contains all the algorithmic implementations used in the simulations.'
'These algorithms include training/testing loops consensus aggregation steps and helper fucntions that implement custom training features for distributed '
'learning like quasi global momentum and adaptive weighting especially selected and used in the experiments.'

'A significant effort has been invested in optimizing the functions for single GPU execution.'
'More optimizations will be included in future updates (if any).'


def DSGD_local_client_update(dev,num_clients, data, client_models,optimizer):

    criterion = nn.CrossEntropyLoss().to(dev) 
    gsd = []
    gpu_data = []
    for inputs_cpu, labels_cpu in data:
        # Move inputs/labels to the device (GPU) efficiently
        inputs_gpu = inputs_cpu.to(dev, non_blocking=True)
        labels_gpu = labels_cpu.to(dev, non_blocking=True)
        gpu_data.append((inputs_gpu, labels_gpu))
 

    
    
    client_grad_tensors = []
    for client_id in range(num_clients):
        client_models[client_id].train()
        #opt = optim.SGD(client_models[client_id].parameters(), lr=lr, momentum = momentum, weight_decay=weight_decay)
        #GUT_set_gradients(client_models[client_index], network_psi[client_index])
        inputs, labels = gpu_data[client_id]
      
        client_models[client_id].zero_grad(set_to_none=True)
        
        # forward + backward + optimize
        outputs = client_models[client_id](inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # 2. Gradient Vectorization Step: Flatten and concatenate all gradients into a single vector will latter be used in the adaptive weighting scheme
        flattened_grad = torch.cat([
            param.grad.view(-1) for param in client_models[client_id].parameters() if param.grad is not None
        ])
        
        client_grad_tensors.append(flattened_grad)
    
        # 3. Final Vectorization Step: Stack all client gradients into a single matrix (SG)
        # The result SG is of shape (num_clients, num_params).
        SG = torch.stack(client_grad_tensors)
        optimizer[client_id].step()
        
       




    return SG











def neighborhood_aggregate(learnables, N, j, A):
    """
    Performs the weighted neighborhood aggregation (consensus) for agent j.
    Optimized to reduce memory copy overhead.

    learnables (list): List of models (or state_dicts) for all agents.
    N (list): List of neighbor lists. N[j] is the list of neighbors for agent j.
    j (int): The index of the agent currently being aggregated (the 'target').
    A (torch.Tensor): The pre-computed weight matrix (adjacency matrix).
    """

    # 1. Start with a deep copy of the target model's state_dict structure.
    #    This copy is necessary because we are modifying the values.
    #    However, we only need to copy the *keys* and set the value to zero.
    
    # Use the target model's state_dict as a template, but initialize values to zero
    # to accumulate the weighted sum.
    target_state_dict = learnables[j].state_dict()
    new_state_dict = {k: torch.zeros_like(v) for k, v in target_state_dict.items()}

    # 2. Iterate through the neighbors of agent j
    for i in N[j]:
        weight = A[j, i].item()
        
        # Get the neighbor's state dict (avoiding deep copy of the whole model list)
        neighbor_state_dict = learnables[i].state_dict()

        # 3. Perform the weighted accumulation
        for k in new_state_dict.keys():
            # In-place addition (faster and saves memory)
            new_state_dict[k].add_(neighbor_state_dict[k], alpha=weight)

    return new_state_dict

def neighborhood_aggregate_sd(state_dicts, N, j, A):
    """
    Performs the weighted neighborhood aggregation for agent j, 
    starting directly from a list of pre-stored state_dicts.

    state_dicts (list): List of state_dicts for all agents (x_0, x_1, ...).
    # ... (other arguments remain the same)
    """
    
    # 1. Start with a zeroed template for the target's state_dict
    #    Target state_dict is now directly state_dicts[j]
    target_state_dict = state_dicts[j] 
    
    # Initialize the new state_dict with zeros (CRITICAL for accumulation)
    new_state_dict = {k: torch.zeros_like(v) for k, v in target_state_dict.items()}

    # 2. Iterate through the neighbors of agent j
    for i in N[j]:
        weight = A[j, i].item()
        
        # Get the neighbor's state dict (DIRECTLY from the list)
        neighbor_state_dict = state_dicts[i] # <--- Change 1: Direct access
        
        # 3. Perform the weighted accumulation
        for k in new_state_dict.keys():
            # In-place addition (fast and saves memory)
            new_state_dict[k].add_(neighbor_state_dict[k], alpha=weight)

    return new_state_dict












def test(dev,num_clients,cmodels, test_loader):

    test_loss = torch.zeros(num_clients,1, device=dev)
    acc = torch.zeros(num_clients,1, device=dev)
    for cmodel in cmodels:
        cmodel.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(dev,non_blocking = True), target.to(dev,non_blocking = True)
            for c_id in range(num_clients):
                cmodel = cmodels[c_id]
                output = cmodel(data)
                batch_loss = F.cross_entropy(output, target, reduction='sum')
                test_loss[c_id] += batch_loss
                
                pred = output.argmax(dim=1, keepdim=True)
                batch_correct = pred.eq(target.view_as(pred)).sum()
                acc[c_id] += batch_correct
    
    test_loss = test_loss.cpu() / len(test_loader.dataset)
    acc = acc.cpu() / len(test_loader.dataset)
    return test_loss, acc
  

def Gombertz_curve(x,alpha):
    G = alpha*(1 - torch.exp(-torch.exp(-alpha*(x-1))))
    return G






def adaptive_combination_weights(dev,st_grad,num_clients,num_proj_params,com_iter,N,stheta, gombertz_alpha):
    #CAN be further optimized for better GPU utilization, using vectorizations.
    #This is a future project!.

    A = torch.zeros(num_clients,num_clients).to(dev)
    TGrad = torch.zeros(num_clients,num_proj_params).to(dev)

    
    
    for i in range(num_clients):
        for j in N[i]:
             TGrad[i,:] += (1/len(N[i]))*(st_grad[j,:])
    

    theta = torch.zeros(num_clients,num_clients).to(dev)
    y = torch.zeros(num_clients,num_clients).to(dev)

    for i in range(num_clients):
        for j in N[i]:
            theta[i,j] = torch.arccos(torch.dot(TGrad[i,:],st_grad[j,:])/(torch.linalg.vector_norm(TGrad[i,:])*(torch.linalg.vector_norm(st_grad[j,:])) + 1e-8))
            #if arccos is instable wrap it in torch.clamp(....,)
   
    # Update smoothed angles
    if com_iter == 0:
        stheta = theta.clone()
    else:
        stheta = ((com_iter-1)/com_iter)*stheta + (1/com_iter)*theta
    for i in range(num_clients):
      for j in N[i]:
        y[i,j] = Gombertz_curve(stheta[i,j],alpha = gombertz_alpha)
        y[i,j] = torch.exp(y[i,j])
        
    y = torch.real(y)
    for i in range(num_clients):
      rsum = torch.sum(y[i,:])
      for j in N[i]:
        A[i,j] = (y[i,j]/rsum).clone()
   
    
    return A,stheta



#INITIALIZE SOME USEFUL VARS DURING RUNNING
def count_model_learnable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters()) 
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return num_params


def initialize_models_buffers(dev,client_models):
    """
    Initializes a list of state dictionary buffers (Zi) by allocating zero-filled 
    tensors on the specified device (GPU) that match the structure of the models.
    """
    Zi_buffers = []
    
    # Iterate through each client model
    for model in client_models:
        # Get the model's state dictionary structure
        model_state = model.state_dict()
        
        # Create a new buffer state dictionary (z_i)
        z_i = {}
        
        # 🎯 Crucial: Initialize a zero tensor for each parameter, 
        #    explicitly specifying the target device (dev).
        for k, v in model_state.items():
            # Use torch.zeros_like to create a tensor with the same size and type,
            # but ensure it is placed on the device (dev).
            # The .to(dev) is often redundant if the source v is already on dev,
            # but it is a safe practice or can be used if v is on CPU.
            z_i[k] = torch.zeros_like(v, device=dev, dtype=v.dtype)

        Zi_buffers.append(z_i)
        
    return Zi_buffers


def update_models_state_dict_in_place(source_models, Zi_buffers):
    """
    Efficiently copies the state_dicts from source_models into the pre-allocated
    tensors in Zi_buffers using in-place operations.
    """
    num_clients = len(source_models)
    
    # 🎯 All operations happen on the GPU (assuming source_models are on the GPU)
    for ind in range(num_clients):
        # 1. Get the current model weights
        source_state = source_models[ind].state_dict()
        
        # 2. Get the pre-allocated destination buffer (for this client)
        dest_state = Zi_buffers[ind]
        
        # 3. Perform an in-place copy (faster than clone)
        for key in source_state.keys():
            # .copy_() performs a fast, in-place copy on the GPU
            # This avoids new memory allocation for dest_state tensors
            dest_state[key].copy_(source_state[key].data)
            
    return Zi_buffers 


def QGm_compute_gradients(dev,num_clients, data, client_models):
    'Computes and gathers all network agents gradients in a state dictionary'
    'These gradients will be used for the quasi-global momentum update.'
    'Also returns the vectorized gradients used in the adaptive weighting step'

    criterion = nn.CrossEntropyLoss().to(dev) 
    gsd = []
    gpu_data = []
    for inputs_cpu, labels_cpu in data:
        # Move inputs/labels to the device (GPU) efficiently
        inputs_gpu = inputs_cpu.to(dev, non_blocking=True)
        labels_gpu = labels_cpu.to(dev, non_blocking=True)
        gpu_data.append((inputs_gpu, labels_gpu))
 


    client_grad_tensors = []
    for client_id in range(num_clients):
        client_models[client_id].train()
        #opt = optim.SGD(client_models[client_id].parameters(), lr=lr, momentum = momentum, weight_decay=weight_decay)
        #GUT_set_gradients(client_models[client_index], network_psi[client_index])
        inputs, labels = gpu_data[client_id]
        inputs = inputs.float()
        labels = labels
        
        client_models[client_id].zero_grad(set_to_none=True)
        
        # forward + backward + optimize
        outputs = client_models[client_id](inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        grad_state_dict = get_gradient_state_dict(client_models[client_id])

        gsd.append(grad_state_dict)
        # 2. Gradient Vectorization Step: Flatten and concatenate all gradients into a single vector will latter be used in the adaptive weighting scheme
        flattened_grad = torch.cat([
            param.grad.view(-1) for param in client_models[client_id].parameters() if param.grad is not None
        ])
        
        client_grad_tensors.append(flattened_grad)
    
        # 3. Final Vectorization Step: Stack all client gradients into a single matrix (SG)
        # The result SG is of shape (num_clients, num_params).
        SG = torch.stack(client_grad_tensors)
        
        


    return gsd,SG

def get_gradient_state_dict(model):
    """
    Extracts the gradients of a PyTorch model into a state dict.

    Args:
        model: The PyTorch model.

    Returns:
        A state dict containing the gradients of the model's parameters.
    """

    gradient_state_dict = OrderedDict()
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradient_state_dict[name] = param.grad.detach().clone()
    
    return gradient_state_dict  


def QGm_momentum_update(num_clients,client_models,opt,grad, M_hat, M, beta):
    #For every agent, compute their leaky average of 
    # quasi-global momentum and update the models.

    for c_id in range(num_clients):
        # M is the target buffer (m_t)
        # M_hat is the previous quasi-global momentum (m_hat_{t-1})
        # grad is the current local gradient (g_t)
        
        for k in grad[c_id].keys():
            # 1. Start with the momentum from M_hat: M[k] = M_hat[k]
            M[c_id][k].copy_(M_hat[c_id][k])
            
            # 2. Apply the beta factor (in-place multiplication): M[k] = M[k] * beta
            M[c_id][k].mul_(beta)
            
            # 3. Add the current gradient (in-place addition): M[k] = M[k] + grad[k]
            M[c_id][k].add_(grad[c_id][k])

        set_gradients(client_models[c_id], M[c_id])
        client_models[c_id].train()
        opt[c_id].step()



def set_gradients(model, gradient_dict):
    """Sets the gradients of a PyTorch model manually from a dictionary.

    Args:
        model: The PyTorch model.
        gradient_dict: A dictionary containing gradient values for each parameter.
    """

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.grad.copy_(gradient_dict[name])
        #param.grad.data.copy_(gradient_dict[name])



def QGm_update_momentum_buffer(num_clients, D, M_hat,client_models, x_t, lrs, beta):
    
    for c_id in range(num_clients):
        # Calculate the scalar multiplier (1/lr)
        # Assuming lrs[c_id] is a PyTorch learning rate scheduler
        lr = lrs[c_id].get_last_lr()[0]
        scalar_factor = 1.0 / lr
        
        # Get the current weight (w_{t+1}) and the weight before update (x_t)
        current_w = client_models[c_id].state_dict()
        prev_w = x_t[c_id]

        for k in prev_w.keys():
            # 1. Start by copying x_t[k] (w_t) into D[k]: D[k] = w_t
            D[c_id][k].copy_(prev_w[k])
            
            # 2. Subtract the current model w_{t+1} (in-place subtraction): D[k] = w_t - w_{t+1}
            D[c_id][k].sub_(current_w[k])
            
            # 3. Apply the learning rate inverse (in-place multiplication): D[k] = D[k] * (1/lr)
            D[c_id][k].mul_(scalar_factor)

            #4. M_hat  = beta*M_hat + (1-beta)*D
            M_hat[c_id][k].mul_(beta)
            M_hat[c_id][k].add_(D[c_id][k], alpha = (1-beta))
            
    return M_hat
