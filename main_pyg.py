import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN

import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

import pdb

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

import sys
sys.path.insert(0,'../..')
from attacks import *


parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
parser.add_argument('--gnn', type=str, default='gcn',
                    help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
parser.add_argument('--drop_ratio', type=float, default=0.5,
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--drop_path_p', type=float, default=0.01,
                    help='droppath ratio (default: 0.01)')
parser.add_argument('--num_layer', type=int, default=5,
                    help='number of GNN message passing layers (default: 5)')
parser.add_argument('--emb_dim', type=int, default=300,
                    help='dimensionality of hidden units in GNNs (default: 300)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                    help='dataset name (default: ogbg-molhiv, ogbg-molpcba)')
parser.add_argument('--feature', type=str, default="full",
                    help='full feature or simple feature')
parser.add_argument('--JK', type=str, default="last",
                    help='full feature or simple feature')

parser.add_argument('--device', type=int, default=0)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--runs', type=int, default=10)

parser.add_argument('--step-size', type=float, default=1e-3)
parser.add_argument('-m', type=int, default=3)
parser.add_argument('--test-freq', type=int, default=1)
parser.add_argument('--attack', type=str, default='flag')

args = parser.parse_args()

def train_vanilla(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def train(model, device, loader, optimizer, task_type, args):
    total_loss = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y

            forward = lambda perturb : model(batch, perturb).to(torch.float32)[is_labeled]
            model_forward = (model, forward)
            y = batch.y.to(torch.float32)[is_labeled]
            perturb_shape = (batch.x.shape[0], args.emb_dim)

            loss, _ = flag(model_forward, perturb_shape, y, args, optimizer, device, cls_criterion)
            total_loss += loss.item()

    return total_loss/len(loader)




def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            for test_iter in range(15):
                pred += model(batch)

            pred = pred/16
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    write_file_name = 'results/result_' 
    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)
    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()
    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)


    vals, tests = [], []
    for run in range(args.runs):
        best_val, final_test = 0, 0

        if args.gnn == 'gin':
            model = GNN(gnn_type='gin', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                        drop_ratio=args.drop_ratio, virtual_node=False).to(device)
        elif args.gnn == 'gin-virtual':
            model = GNN(gnn_type='gin', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                        drop_ratio=args.drop_ratio, virtual_node=True).to(device)
        elif args.gnn == 'gcn':
            model = GNN(gnn_type='gcn', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                        drop_ratio=args.drop_ratio, virtual_node=False).to(device)
        elif args.gnn == 'gcn-virtual':
            model = GNN(gnn_type='gcn', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                        drop_ratio=args.drop_ratio, virtual_node=True).to(device)
        elif args.gnn == 'randomgin':
            model = GNN(gnn_type = 'randomgin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, drop_path_p=args.drop_path_p, virtual_node = False).to(device)
        elif args.gnn == 'randomgin-virtual':
            model = GNN(gnn_type = 'randomgin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, JK = args.JK, drop_ratio = args.drop_ratio, drop_path_p=args.drop_path_p, virtual_node = True).to(device)
        else:
            raise ValueError('Invalid GNN type')

        tot_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("No. params: %d" % (tot_params, ))

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


        for epoch in range(1, args.epochs+1):
            print("=====Epoch {}".format(epoch))
            print('Training...')
            loss = train(model, device, train_loader, optimizer, dataset.task_type, args)
            if epoch > args.epochs//2 and epoch % args.test_freq == 0 or epoch == args.epochs:
                print('Evaluating...')
                train_perf = eval(model, device, train_loader, evaluator)
                valid_perf = eval(model, device, valid_loader, evaluator)
                test_perf = eval(model, device, test_loader, evaluator)
                print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
                result = (train_perf[dataset.eval_metric], valid_perf[dataset.eval_metric], test_perf[dataset.eval_metric])
                _, val, tst = result
                if val > best_val:
                    best_val = val
                    final_test = tst
            if epoch == 1:
                print('Evaluating...')
                train_perf = eval(model, device, train_loader, evaluator)
                valid_perf = eval(model, device, valid_loader, evaluator)
                test_perf = eval(model, device, test_loader, evaluator)
                print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        print(f'Run{run} val:{best_val}, test:{final_test}')
        with open(write_file_name + '_' + args.JK + '_run' + str(run) + '.txt', 'w') as f:
        	f.write("""Run: {}\nVal {:.4f}\nTest: {:.4f}\n\n\n""".format(run, best_val, final_test))
        vals.append(best_val)
        tests.append(final_test)
        

    print('')
    print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals)}")
    print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests)}")

if __name__ == "__main__":
    main()
