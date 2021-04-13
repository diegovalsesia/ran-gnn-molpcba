# ogbg-molpcba

To train baselines with FLAG in the default setup, run

**GIN+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol).

    python main_pyg.py --dataset ogbg-molpcba --gnn gin --step-size 8e-3

**GIN+V+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol).

    python main_pyg.py --dataset ogbg-molpcba --gnn gin-virtual --step-size 8e-3

To train baselines with FLAG and random architecture in the default setup, run

**RAN-GIN+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol).

    python main_pyg.py --dataset ogbg-molpcba --gnn randomgin --step-size 8e-3 --num_layer 12 --emb_dim 200

**RAN-GIN+V+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol).

    python main_pyg.py --dataset ogbg-molpcba --gnn randomgin-virtual --step-size 8e-3 --num_layer 12 --emb_dim 248


THis code is derived from https://github.com/devnkong/FLAG


