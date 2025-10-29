'''
Train an S4 model on sequential CIFAR10 / sequential MNIST with PyTorch for demonstration purposes.
This code borrows heavily from https://github.com/kuangliu/pytorch-cifar.

This file only depends on the standalone S4 layer
available in /models/s4/

* Train standard sequential CIFAR:
    python -m example
* Train sequential CIFAR grayscale:
    python -m example --grayscale
* Train MNIST:
    python -m example --dataset mnist --d_model 256 --weight_decay 0.0

The `S4Model` class defined in this file provides a simple backbone to train S4 models.
This backbone is a good starting point for many problems, although some tasks (especially generation)
may require using other backbones.

The default CIFAR10 model trained by this file should get
89+% accuracy on the CIFAR10 test set in 80 epochs.

Each epoch takes approximately 7m20s on a T4 GPU (will be much faster on V100 / A100).
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial

import os
import argparse

from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from models.s4.s4d import S4D
from dataset import TimeseriesDataset

from tqdm.auto import tqdm

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Optimizer
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
# Scheduler
# parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs', default=10, type=float, help='Training epochs')
# Dataset
parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
parser.add_argument('--model', default='s4', choices=['s4', 's4d'], type=str)
parser.add_argument('--context_length', default=864, type=int)
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_val_loss = float('inf') # best validation loss
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print(f'==> Preparing data..')

timeseries = torch.load('timeseries.pt')
tile_features = torch.load('tile_ft.pt')
time_covariates = torch.load('time_ft.pt')

val_split = 0.1
d_input = 1 + tile_features.shape[1] + time_covariates.shape[1]

time_len = int(timeseries.shape[1] * (1 - val_split))
timeseries_train, timeseries_val = timeseries[:,:time_len], timeseries[:,time_len:]
time_covariates_train, time_covariates_val = time_covariates[:time_len], time_covariates[time_len:]

trainset = TimeseriesDataset(
    data=timeseries_train, 
    tile_features=tile_features, 
    time_covariates=time_covariates_train,
    context_length=args.context_length
)
valset = TimeseriesDataset(
    data=timeseries_val, 
    tile_features=tile_features, 
    time_covariates=time_covariates_val,
    context_length=args.context_length
)

# Dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

class DeepARS4(nn.Module):

    def __init__(
        self,
        d_input: int,
        d_model: int = 256,
        n_layers: int = 4,
        dropout: float = 0.2,
        prenorm: bool = False,
    ):
        """
        Args:
            d_input: Input dimension of model. It should be set to 1+(#features), 1 for the last observed value + features for the current timestep.
            d_model: Model dimension, or number of independent convolution kernels created
            n_layers: How many S4/S4D layers to stack
            dropout: Dropout
            prenorm: True if normalization should happen before applying the S4 layer. This is what newer models use today. The original transformer used postnorm where the residual input also gets normalized.
        """
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (1 + #features)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            assert args.model in ['s4', 's4d'], "Invalid args.model"
            self.s4_layers.append(
                S4(d_model=d_model, bidirectional=False, l_max=1024, final_act="glu", dropout=dropout, transposed=False)
                if args.model == 's4' else
                S4D(d_model, dropout=dropout, transposed=False, lr=min(0.001, args.lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder_mu = nn.Linear(d_model, 1)
        self.decoder_alpha = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, L, d_model) -> (B, L, d_model)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x)

        # Decode the outputs
        mu = F.softplus(self.decoder_mu(x))  # (B, L, d_model) -> (B, L, 1)
        alpha = F.softplus(self.decoder_alpha(x))  # (B, L, d_model) -> (B, L, 1)

        return mu, alpha

class NegativeBinomialNLL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, alpha, targets):
        return (
            -NegativeBinomial(
                total_count=1.0 / alpha, 
                probs= 1.0 - 1.0 / (1.0 + alpha*mu)
            ).log_prob(targets).mean()
        )

# Model
print('==> Building model..')
model = DeepARS4(
    d_input=d_input,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm,
)

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_val_loss = checkpoint['avg_loss']
    start_epoch = checkpoint['epoch']

def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

criterion = NegativeBinomialNLL()
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train():
    model.train()
    train_loss = 0
    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        mu, alpha = model(inputs)
        loss = criterion(mu, alpha, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | MAE: %.3f' %
            (batch_idx, len(trainloader), train_loss/(batch_idx+1), torch.mean(torch.abs(mu - targets)))
        )


def eval(epoch, dataloader, checkpoint=False):
    global best_val_loss
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            mu, _ = outputs
            loss = criterion(outputs, targets)

            eval_loss += loss.item()

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | MAE: %.3f' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1), torch.mean(torch.abs(mu - targets)))
            )

    avg_eval_loss = eval_loss / len(dataloader)

    # Save checkpoint.
    if checkpoint:
        if avg_eval_loss < best_val_loss:
            state = {
                'model': model.state_dict(),
                'avg_loss': avg_eval_loss,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_val_loss = avg_eval_loss

        return avg_eval_loss

avg_eval_loss = 0
pbar = tqdm(range(start_epoch, args.epochs))
for epoch in pbar:
    if epoch == 0:
        pbar.set_description('Epoch: %d' % (epoch))
    else:
        pbar.set_description('Epoch: %d | Avg eval loss: %.3f' % (epoch, avg_eval_loss))
    train()
    avg_eval_loss = eval(epoch, valloader, checkpoint=True)
    scheduler.step()
    # print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")

