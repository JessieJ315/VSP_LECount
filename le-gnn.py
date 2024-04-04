## helper functions

def get_edge_list(tc)->torch.tensor:
  '''Obtain edge list given a transitive closure.'''
  indices = np.where(tc == 1)
  edges = torch.tensor([indices[0],indices[1]])
  return edges

def data_generation(max_num_nodes, num_features, num_graphs, min_num_nodes=3):
  '''Generates list[torch_geometric.data.Data] objects.'''
  graphs = []
  for i in range(num_graphs):
    num_nodes = max_num_nodes # random.randint(min_num_nodes,max_num_nodes)
    actors = np.arange(start=i*num_nodes,stop=(i+1)*num_nodes)
    tree = vspfun.random_tree(actors)
    tr = pofun.transitive_reduction(vspfun.tree2vsp(tree))
    # edges = get_edge_list(tc)
    le = vspfun.nle_tree(tree)
    log_le = -np.log(float(le))
    node_feature = torch.ones((num_nodes,num_features),dtype=torch.float32)
    graph = Data(y=torch.tensor(log_le,dtype=torch.float32),x=node_feature,adj=torch.tensor(tr,dtype=torch.float32))#,edge_index=edges)
    graphs.append(graph)
  return graphs

## GNN class

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes = ceil(0.25 * n)
        self.gnn1_pool = GNN(num_features, 64, num_nodes)
        self.gnn1_embed = GNN(num_features, 64, 64, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, 1)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x, l1 + l2, e1 + e2
