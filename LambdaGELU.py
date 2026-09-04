import torch
import numpy as np

class LambdaGELU(torch.nn.Module):
    '''
    Lambda-GELU Activation Function with Learnable Gating Hardness
    Reference: Pérez-Corral et al., "lambda-GELU: Learning Gating Hardness for Controlled ReLU-ization"[cite: 1]
    '''
    def __init__(self, s_param: float = 0.0, temperature: float = 0.1):
        super(LambdaGELU, self).__init__()
        self.s_param = torch.nn.Parameter(torch.tensor(s_param, dtype=torch.float32))
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Temperature-controlled softplus reparameterization: lambda(s) = 1 + softplus(s / t)[cite: 1]
        lam = 1.0 + torch.nn.functional.softplus(self.s_param / self.temperature)
        
        # f(x; lambda) = x * Phi(lambda * x), where Phi is the standard normal CDF[cite: 1]
        output = 0.5 * x * (1.0 + torch.erf((lam * x) / np.sqrt(2.0)))
        return output