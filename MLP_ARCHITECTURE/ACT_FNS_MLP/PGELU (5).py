
import torch as torch
import math

class PGELU(torch.nn.Module):
    '''
    Parametric Gaussian Error Linear Unit (PGELU) Activation Function.

    Implementation Author: Jose Aries E. De Los Santos

    This implementation is based on the mathematical formulation proposed in:
    Labied, M., Belangour, A., & Banane, M. (2025). P-GELU: A novel activation 
    function to optimize Whisper for Darija speech translation. IEEE Access, 
    13, 100198-100218. https://doi.org/10.1109/ACCESS.2025.3574398

    Note: The original paper contains a typographical error in Equation 7, 
    writing the function as `tan`. This implementation correctly uses `tanh` 
    to prevent gradient explosion and maintain the intended smooth, continuous curve[cite: 1].
    '''
    def __init__(self,
                 alpha_param: float = 1.0,
                 beta_param: float = 0.04):
        super(PGELU, self).__init__()
        self.alpha_param = torch.nn.Parameter(torch.tensor(alpha_param, dtype=torch.float32))
        self.beta_param = torch.nn.Parameter(torch.tensor(beta_param, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.mul(x , (1 + torch.tanh(torch.mul(self.alpha_param, x) + torch.mul(self.beta_param, torch.pow(x, 3)))))
        return output
        