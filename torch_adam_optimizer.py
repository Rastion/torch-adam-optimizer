import torch
from rastion_core.base_optimizer import BaseOptimizer

class TorchAdamOptimizer(BaseOptimizer):
    """
    A classical optimizer that uses PyTorch's Adam optimizer to update
    the variational parameters. This class provides a step_and_cost method
    that is compatible with the vqa_cycle.
    """
    def __init__(self, lr=0.01):
        self.lr = lr
        self.optimizer = None
        self.theta = None

    def step_and_cost(self, cost_function, theta):
        """
        Perform one optimization step on theta using the cost_function.
        If this is the first call, the optimizer is initialized.
        
        Parameters:
          - cost_function: a function that takes a torch tensor theta and returns a scalar cost.
          - theta: a torch.tensor (with requires_grad=True)
        
        Returns:
          - updated theta (detached)
          - current cost (as a float)
        """
        # Initialize and store the parameter and optimizer only once.
        if self.theta is None:
            self.theta = theta
            self.optimizer = torch.optim.Adam([self.theta], lr=self.lr)
        # Zero out gradients, compute cost and backpropagate.
        self.optimizer.zero_grad()
        cost = cost_function(self.theta)
        cost.backward()
        self.optimizer.step()
        return self.theta.detach(), cost.item()
    
    def optimize(self, problem, **kwargs):
        """
        Dummy implementation of the abstract method.
        Since we use this optimizer only for its step_and_cost method,
        this method is not intended to be used.
        """
        raise NotImplementedError("Use step_and_cost() instead of optimize() for TorchAdamOptimizer.")
