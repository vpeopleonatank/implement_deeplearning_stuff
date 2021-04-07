import torch
import torch.nn as nn

def main():
    outputs = torch.randn((1, 5))
    targets = torch.tensor([1,1,1,1,0])
    criterion = nn.BCELoss()
    loss = criterion(outputs, targets)
    print(loss.item())



if __name__ == '__main__':
    main()

