
import torch as torch
import numpy as np

class MPGELU(torch.nn.Module):
    '''
    Modified Parametric Gaussian Error Linear Unit (MPGELU) Activation Function
    '''
    def __init__(self,s_param: float = 0.0, use_softplus: bool = True):
        super(MPGELU, self).__init__()
        self.s_param = torch.nn.Parameter(torch.tensor(s_param, dtype=torch.float32))
        self.use_softplus = use_softplus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_softplus:
            lam = 1.0 + torch.nn.functional.softplus(self.s_param)
        else:
            lam = 1.0 + torch.log(torch.tensor(1.0, device=x.device) + torch.exp(self.s_param))
        output = torch.mul(0.5 * x, (1.0 + torch.erf(torch.mul(lam,x)/ (np.sqrt(2)))))
        return output
        

