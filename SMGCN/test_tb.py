import torch 
import torch.nn as nn
from torch.autograd import Variable
import tensorboardX as tb 


class Net(nn.Module):
    def __init__(self, dim1, dim2):
        super(Net, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(dim1, dim2), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.randn(dim2), requires_grad=True)

    def forward(self, x, y):
        # assert x.size(0) == y.size(0)
        res1 = torch.mm(x, self.weight) + self.bias
        res2 = torch.mm(y, self.weight) + self.bias
        return torch.add(res1, res2)


if __name__ == "__main__":
    model = Net(34, 20)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(),lr=1e-3)
    label = torch.randn(12,20)
    input1 = torch.randn(12,34)
    input2 = torch.randn(12, 34)

    writer = tb.SummaryWriter('extra/', comment = 'Net')
    with writer:
        writer.add_graph(model, (input1, input2))

        

    # for module in model.named_modules():
    #     print(module)
    #     # print('name:{}, module:{}'.format(name, module))










