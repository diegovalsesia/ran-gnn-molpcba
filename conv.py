import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree
from utils_random import Node, get_graph_info, build_graph, save_graph, load_graph

import math
import pdb

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        #self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.mlp1 = torch.nn.Linear(emb_dim, 2*emb_dim)
        self.mlp2 = torch.nn.Linear(2*emb_dim, emb_dim)
        self.norm = nn.BatchNorm1d(2*emb_dim)
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.relu = nn.ReLU()
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)    
        aa = self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        bb = (1 + self.eps) *x + aa
        out = self.mlp1(bb)
        out = self.norm(out)
        out = self.relu(out)
        out = self.mlp2(out)
        #out = self.mlp(bb)
    
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GIN_Triplet_unit(nn.Module):
  
  def __init__(self, emb_dims, drop_ratio):
    super(GIN_Triplet_unit, self).__init__()
    self.relu = nn.ReLU()
    self.conv = GINConv(emb_dims)
    self.drop_ratio = drop_ratio
    self.norm = nn.BatchNorm1d(emb_dims)

  def forward(self, h, e_i, e_a, add_activation):
    #out = self.relu(h) # to be checked
    out = self.conv(h, e_i, e_a)
    out = self.norm(out)
    if add_activation:
        out = self.relu(out)
    out = F.dropout(out, self.drop_ratio, training = self.training)
    return out


class Node_OP(nn.Module):

  def __init__(self, Node, emb_dims, drop_ratio, drop_path_p, add_virtualnode = False):
    super(Node_OP, self).__init__()
    self.is_input_node = Node.type == 0
    self.input_nums = len(Node.inputs)
    self.add_virtualnode = add_virtualnode
    self.drop_path_p = drop_path_p
    if self.input_nums > 1:
        self.mean_weight = nn.Parameter(torch.ones(self.input_nums))
        self.sigmoid = nn.Sigmoid()
    if self.add_virtualnode:
        self.mean_weight_vn = nn.Parameter(torch.ones(self.input_nums))
        self.sigmoid_vn = nn.Sigmoid()
        self.mlp_virtualnode = torch.nn.Sequential(torch.nn.Linear(emb_dims, 2*emb_dims), torch.nn.BatchNorm1d(2*emb_dims), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dims, emb_dims), torch.nn.BatchNorm1d(emb_dims), torch.nn.ReLU())
    self.conv = GIN_Triplet_unit(emb_dims, drop_ratio)
    self.drop_ratio = drop_ratio

  def forward(self, *input):
    if self.input_nums > 1:     
        if torch.rand(1) < self.drop_path_p:
            out = 0.0*input[0][0]
            if self.add_virtualnode:
                vn_aggr = 0.0*input[0][4]
        else:
            out = self.sigmoid(self.mean_weight[0]) * input[0][0]
            if self.add_virtualnode:
                vn_aggr = self.sigmoid_vn(self.mean_weight_vn[0]) * input[0][4]
            #out = self.mean_weight[0] * input[0][1]
        for i in range(1, self.input_nums):
            if torch.rand(1) > self.drop_path_p:
                #aa = self.sigmoid(self.mean_weight[i]) * input[i][0]

                out = out + self.sigmoid(self.mean_weight[i]) * input[i][0]
                if self.add_virtualnode:
                    vn_aggr = vn_aggr + self.sigmoid(self.mean_weight_vn[i]) * input[i][4]
                #out = out + self.mean_weight[i] * input[i][1]   
        if self.add_virtualnode:
            out = out + vn_aggr[input[0][5]]
    else:
        if self.add_virtualnode:
            out = input[0][0] + input[0][4][input[0][5]]
            vn_aggr = 1.0*input[0][4]
        else:
            out = input[0][0]
    if self.add_virtualnode:
        virtualnode_embedding_temp = global_add_pool(out, input[0][5]) + vn_aggr
        virtualnode_embedding = F.dropout(self.mlp_virtualnode(virtualnode_embedding_temp), self.drop_ratio, training = self.training)

    out = self.conv(out, input[0][1], input[0][2], input[0][3])
    
    if self.add_virtualnode:
        return out, virtualnode_embedding
    else:
        return out



class StageBlock(nn.Module):

  def __init__(self, net_graph, emb_dim, net_linear, drop_ratio, drop_path_p = 0.0, add_virtualnode = False):
    super(StageBlock, self).__init__()
    self.nodes, self.input_nodes, self.output_nodes = get_graph_info(net_graph, net_linear)
    self.nodeop  = nn.ModuleList()
    self.add_virtualnode = add_virtualnode
    for node in self.nodes:
      self.nodeop.append(Node_OP(node, emb_dim, drop_ratio, drop_path_p, add_virtualnode))
    

  def forward(self, h, e_i, e_a,virtualnode_embedding = None, batch = None):
    results = {}
    virtualnode = {}
    for id in self.input_nodes:
        if self.add_virtualnode:
            results[id], virtualnode[id] = self.nodeop[id](*[[h, e_i, e_a, True, virtualnode_embedding, batch]])
        else:
            results[id] = self.nodeop[id](*[[h, e_i, e_a, True]])
    for id, node in enumerate(self.nodes):
        if id not in self.input_nodes:
            if id not in self.output_nodes:
                if self.add_virtualnode:
                    results[id], virtualnode[id] = self.nodeop[id](*[[results[_id], e_i, e_a, True, virtualnode[_id], batch] for _id in node.inputs])
                else:
                    results[id] = self.nodeop[id](*[[results[_id], e_i, e_a, True] for _id in node.inputs])
            else:
                if self.add_virtualnode:
                    results[id], virtualnode[id] = self.nodeop[id](*[[results[_id], e_i, e_a, False, virtualnode[_id], batch] for _id in node.inputs])
                else:
                    results[id] = self.nodeop[id](*[[results[_id], e_i, e_a, False] for _id in node.inputs])
    result = results[self.output_nodes[0]]
    for idx, id in enumerate(self.output_nodes):
        if idx > 0:
            result = result + results[id]
    result = result / len(self.output_nodes)
    return results


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))
                
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding

        tmp = self.atom_encoder(x) + perturb if perturb is not None else self.atom_encoder(x)
        h_list = [tmp]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]

        return node_representation


### GNN to generate node embedding
class RandomGNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin', drop_path_p = 0.01 , net_linear = False, net_seed = 47, edge_p = 0.6):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(RandomGNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.net_linear = net_linear
        self.drop_path_p = drop_path_p
        self.net_args={'graph_model' : 'ER',
                       'P': edge_p,
                       'seed': net_seed,
                       'net_linear': self.net_linear}
        net_graph = build_graph(self.num_layer-1, self.net_args)
        self.stage = StageBlock(net_graph, emb_dim, self.net_linear, self.drop_ratio, self.drop_path_p)


        # for layer in range(num_layer):
        # #     if gnn_type == 'gin':
        # #         self.convs.append(GINConv(emb_dim))
        # #     elif gnn_type == 'gcn':
        # #         self.convs.append(GCNConv(emb_dim))
        # #     else:
        # #         ValueError('Undefined GNN type called {}'.format(gnn_type))
                
        #     self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch


        ### computing input node embedding

        h = self.atom_encoder(x) + perturb if perturb is not None else self.atom_encoder(x)

        h_list = [h]

        h_stage = self.stage(h, edge_index, edge_attr)

        for layer in h_stage:

            hh = h_stage[layer]#self.convs[layer](h_list[layer], edge_index, edge_attr)
            # h = self.batch_norms[layer](h)

            # if layer == self.num_layer - 1:
            #     #remove relu for the last layer
            #     h = F.dropout(h, self.drop_ratio, training = self.training)
            # else:
            #     h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            # if self.residual:
            #     h += h_list[layer]

            h_list.append(hh)


        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]
        

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))


    def forward(self, batched_data, perturb=None):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        tmp = self.atom_encoder(x) + perturb if perturb is not None else self.atom_encoder(x)
        h_list = [tmp]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]
        
        return node_representation


class RandomGNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin', drop_path_p = 0.01 , net_linear = False, net_seed = 47, edge_p = 0.6):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(RandomGNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.drop_path_p = drop_path_p
        self.net_linear = net_linear
        self.net_args={'graph_model' : 'ER',
                       'P': edge_p,
                       'seed': net_seed,
                       'net_linear': self.net_linear}
        net_graph = build_graph(self.num_layer-1, self.net_args)
        self.stage = StageBlock(net_graph, emb_dim, self.net_linear, self.drop_ratio, self.drop_path_p, True)


        # for layer in range(num_layer):
        # #     if gnn_type == 'gin':
        # #         self.convs.append(GINConv(emb_dim))
        # #     elif gnn_type == 'gcn':
        # #         self.convs.append(GCNConv(emb_dim))
        # #     else:
        # #         ValueError('Undefined GNN type called {}'.format(gnn_type))
                
        #     self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        ### computing input node embedding

        h = self.atom_encoder(x) + perturb if perturb is not None else self.atom_encoder(x)

        h_list = [h]

        h_stage = self.stage(h, edge_index, edge_attr, virtualnode_embedding, batch)

        for layer in h_stage:

            hh = h_stage[layer]#self.convs[layer](h_list[layer], edge_index, edge_attr)
            # h = self.batch_norms[layer](h)

            # if layer == self.num_layer - 1:
            #     #remove relu for the last layer
            #     h = F.dropout(h, self.drop_ratio, training = self.training)
            # else:
            #     h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            # if self.residual:
            #     h += h_list[layer]

            h_list.append(hh)


        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]
        

        return node_representation


if __name__ == "__main__":
    pass