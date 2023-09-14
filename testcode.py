import argparse
import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser("Continuous Normalizing Flow")
parser.add_argument("--hidden_size", type=int, default=4)
parser.add_argument("--hidden_layer", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-5)
args = parser.parse_args()
print(f"cuda: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


input_size = 10
output_size = 10
model_lst = [nn.Linear(input_size, args.hidden_size)]
for i in range(args.hidden_layer):
      model_lst.append(nn.Linear(args.hidden_size,args.hidden_size))
model_lst.append(nn.Linear(args.hidden_size,output_size))


model = nn.Sequential(*model_lst).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 


for itr in range(200):
      x = torch.randn(args.batch_size, input_size).to(device) 
      y = torch.randn(args.batch_size, output_size).to(device) 
      y_hat = model(x)
      loss = (y - y_hat).pow(2).mean()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      print(f"{itr}: {loss.item():.3f}")
