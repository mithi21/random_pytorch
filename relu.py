import torch
import torch.nn as nn

class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()

    def forward(self, x):
        return torch.where(x > 0, x, torch.zeros_like(x))
        

if __name__ == "__main__":
    x = torch.randn(1, 3, 5, 5)
    relu = CustomReLU()
    custom_output = relu(x)
    pt_relu = nn.ReLU()
    pytorch_output = pt_relu(x)
    output_match = torch.allclose(custom_output, pytorch_output.detach(), atol=1e-6)
    print(output_match)
    assert output_match, "The outputs do not match"
    