# Task04 数据完整存储与内存的数据集类& 节点预测与边预测实践

## 一、数据完全存储于内存的数据集

* **目标：实现在PyG中自定义一个数据完全存于内存的数据集**

  * 通过继承[`InMemoryDataset`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset)类来自定义一个数据可全部存储到内存的数据集类

  ```python
  class InMemoryDataset(root: Optional[str] = None, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None)
  ```

  `InMemoryDataset`官方文档：[`torch_geometric.data.InMemoryDataset`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset)

  如上方的[`InMemoryDataset`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset)类的构造函数接口所示，每个数据集都要有一个**根文件夹（`root`）**，它指示数据集应该被保存在哪里。在根目录下至少有两个文件夹：

  - 一个文件夹为**`raw_dir`**，它用于存储未处理的文件，从网络上下载的数据集文件会被存放到这里；
  - 另一个文件夹为**`processed_dir`**，处理后的数据集被保存到这里。

  此外，继承[`InMemoryDataset`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset)类的每个数据集类可以传递一个**`transform`函数**，一个**`pre_transform`函数**和一个**`pre_filter`**函数，它们默认都为`None`。

  - **`transform`**函数接受`Data`对象为参数，对其转换后返回。此函数在每一次数据访问时被调用，所以它应该用于数据增广（Data Augmentation）。
  - **`pre_transform`**函数接受 [`Data`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data)对象为参数，对其转换后返回。此函数在样本 [`Data`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data)对象保存到文件前调用，所以它最好用于只需要做一次的大量预计算。
  - **`pre_filter`**函数可以在保存前手动过滤掉数据对象。该函数的一个用例是，过滤样本类别。

  为了创建一个[`InMemoryDataset`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset)，需要**实现四个基本方法**：

  - [**`raw_file_names()`**](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset.raw_file_names)。这是一个属性方法，返回一个文件名列表，文件应该能在`raw_dir`文件夹中找到，否则调用`process()`函数下载文件到`raw_dir`文件夹。

  - [**`processed_file_names()`**](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset.processed_file_names)。这是一个属性方法，返回一个文件名列表，文件应该能在`processed_dir`文件夹中找到，否则调用`process()`函数对样本做预处理然后保存到`processed_dir`文件夹。

  - [**`download()`**](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset.download): 将原始数据文件下载到`raw_dir`文件夹。

  - [**`process()`**](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset.process): 对样本做预处理然后保存到`processed_dir`文件夹。

    ```python
    ## 继承InMemoryDataset并实现以上四个方法
    import torch
    from torch_geometric.data import InMemoryDataset, download_url
    
    class MyOwnDataset(InMemoryDataset):
        def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
            super().__init__(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])
    
        @property
        def raw_file_names(self):
            return ['some_file_1', 'some_file_2', ...]
    
        @property
        def processed_file_names(self):
            return ['data.pt']
    
        def download(self):
            # Download to `self.raw_dir`.
            download_url(url, self.raw_dir)
            ...
    
        def process(self):
            # Read data into huge `Data` list.
            data_list = [...]
    
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
    
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
    
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
    
    ```

    `process`函数：样本从原始文件转换成 [`Data`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data)类对象的过程。在该函数中，有时我们需要读取和创建一个 [`Data`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data)对象的列表，并将其保存到`processed_dir`中。由于python保存一个巨大的列表是相当慢的，因此我们在保存之前通过[`collate()`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset.collate)函数将该列表集合成一个巨大的 [`Data`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data)对象。该函数还会返回一个切片字典，以便从这个对象中重构单个样本。最后，我们需要在构造函数中把这`Data`对象和切片字典分别加载到属性`self.data`和`self.slices`中。

### 实例：生成一个[`InMemoryDataset`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset)子类对象时程序的运行流程

以公开数据集`PubMed`为例子。`PubMed `数据集存储的是文章引用网络，文章对应图的结点，如果两篇文章存在引用关系（无论引用与被引），则这两篇文章对应的结点之间存在边。该数据集来源于论文[Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/pdf/1603.08861.pdf)。我们直接基于PyG中的`Planetoid`类修改得到下面的`PlanetoidPubMed`数据集类。


```python
import os.path as osp

import torch
from torch_geometric.data import (InMemoryDataset, download_url)
from torch_geometric.io import read_planetoid_data

class PlanetoidPubMed(InMemoryDataset):
    r"""The citation network datasets "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): The type of dataset split
            (:obj:`"public"`, :obj:`"full"`, :obj:`"random"`).
            If set to :obj:`"public"`, the split will be the public fixed split
            from the
            `"Revisiting Semi-Supervised Learning with Graph Embeddings"
            <https://arxiv.org/abs/1603.08861>`_ paper.
            If set to :obj:`"full"`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"random"` split. (default: :obj:`20`)
        num_val (int, optional): The number of validation samples in case of
            :obj:`"random"` split. (default: :obj:`500`)
        num_test (int, optional): The number of test samples in case of
            :obj:`"random"` split. (default: :obj:`1000`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root, split="public", num_train_per_class=20,
                 num_val=500, num_test=1000, transform=None,
                 pre_transform=None):

        super(PlanetoidPubMed, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.split = split
        assert self.split in ['public', 'full', 'random']

        if split == 'full':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return ['ind.pubmed.{}'.format(name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, 'pubmed')
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

```

在我们生成一个`PlanetoidPubMed`类的对象时，**程序运行流程**如下：

- 首先**检查数据原始文件是否已下载**：
  - 检查`self.raw_dir`目录下是否存在`raw_file_names()`属性方法返回的每个文件，
  - 如有文件不存在，则调用`download()`方法执行原始文件下载。
  - 其中`self.raw_dir`为`osp.join(self.root, 'raw')`。
- 其次**检查数据是否经过处理**：
  - 首先**检查之前对数据做变换的方法**：检查`self.processed_dir`目录下是否存在`pre_transform.pt`文件：如果存在，意味着之前进行过数据变换，则需加载该文件获取之前所用的数据变换的方法，并检查它与当前`pre_transform`参数指定的方法是否相同；如果不相同则会报出一个警告，“The pre_transform argument differs from the one used in ……”。
  - 接着**检查之前的样本过滤的方法**：检查`self.processed_dir`目录下是否存在`pre_filter.pt`文件，如果存在，意味着之前进行过样本过滤，则需加载该文件获取之前所用的样本过滤的方法，并检查它与当前`pre_filter`参数指定的方法是否相同，如果不相同则会报出一个警告，“The pre_filter argument differs from the one used in ……”。其中`self.processed_dir`为`osp.join(self.root, 'processed')`。
  - 接着**检查是否存在处理好的数据**：检查`self.processed_dir`目录下是否存在`self.processed_paths`方法返回的所有文件，如有文件不存在，意味着不存在已经处理好的样本的文件，如需执行以下的操作：
    - 调用`process`方法，进行数据处理。
    - 如果`pre_transform`参数不为`None`，则调用`pre_transform`方法进行数据处理。
    - 如果`pre_filter`参数不为`None`，则进行样本过滤（此例子中不需要进行样本过滤，`pre_filter`参数始终为`None`）。
    - 保存处理好的数据到文件，文件存储在`processed_paths()`属性方法返回的路径。如果将数据保存到多个文件中，则返回的路径有多个。这些路径都在`self.processed_dir`目录下，以`processed_file_names()`属性方法的返回值为文件名。
    - 最后保存新的`pre_transform.pt`文件和`pre_filter.pt`文件，其中分别存储当前使用的数据处理方法和样本过滤方法。


```python
dataset = PlanetoidPubMed('../dataset/Planetoid/PubMed')
print(dataset.num_classes)
print(dataset[0].num_nodes)
print(dataset[0].num_edges)
print(dataset[0].num_features)

# 3
# 197171
# 88648
# 500
```

    3
    19717
    88648
    500


可以看到这个数据集包含三个分类任务，共19,717个结点，88,648条边，节点特征维度为500。

## 二、节点预测与边预测实践

* 目标：掌握实际应用-节点预测或边预测问题
### 2.1 节点预测实践

重定义一个GAT神经网络，使其能够通过参数定义`GATConv`的层数，以及每一层`GATConv`的`out_channels`


```python
from torch_geometric.nn import GATConv, Sequential
from torch.nn import Linear, ReLU
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels_list, num_classes):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        hns = [num_features] + hidden_channels_list
        conv_list = []
        for idx in range(len(hidden_channels_list)):
            conv_list.append((GATConv(hns[idx], hns[idx+1]), 'x, edge_index -> x'))
            conv_list.append(ReLU(inplace=True),)

        self.convseq = Sequential('x, edge_index', conv_list)
        self.linear = Linear(hidden_channels_list[-1], num_classes)

    def forward(self, x, edge_index):
        x = self.convseq(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        return x

```


```python
def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    # Compute the loss solely based on the training nodes.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc

```


```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
%matplotlib inline

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color.cpu(), cmap="Set2")
    plt.show()

```


```python
from torch_geometric.transforms import NormalizeFeatures

dataset = PlanetoidPubMed(root='dataset/PlanetoidPubMed/', transform=NormalizeFeatures())
print('dataset.num_features:', dataset.num_features)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)

model = GAT(num_features=dataset.num_features, hidden_channels_list=[200, 100], num_classes=dataset.num_classes).to(
    device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 201):
    loss = train()
    if epoch % 10 == 0: 
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')

model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y)
```

    dataset.num_features: 500
    GAT(
      (convseq): Sequential(
        (0): GATConv(500, 200, heads=1)
        (1): ReLU(inplace=True)
        (2): GATConv(200, 100, heads=1)
        (3): ReLU(inplace=True)
      )
      (linear): Linear(in_features=100, out_features=3, bias=True)
    )
    Epoch: 010, Loss: 0.7774
    Epoch: 020, Loss: 0.1935
    Epoch: 030, Loss: 0.0492
    Epoch: 040, Loss: 0.0189
    Epoch: 050, Loss: 0.0156
    Epoch: 060, Loss: 0.0214
    Epoch: 070, Loss: 0.0089
    Epoch: 080, Loss: 0.0123
    Epoch: 090, Loss: 0.0113
    Epoch: 100, Loss: 0.0203
    Epoch: 110, Loss: 0.0091
    Epoch: 120, Loss: 0.0139
    Epoch: 130, Loss: 0.0192
    Epoch: 140, Loss: 0.0136
    Epoch: 150, Loss: 0.0175
    Epoch: 160, Loss: 0.0081
    Epoch: 170, Loss: 0.0046
    Epoch: 180, Loss: 0.0075
    Epoch: 190, Loss: 0.0083
    Epoch: 200, Loss: 0.0062
    Test Accuracy: 0.7600




![png](https://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/output_9_1.png)
    


## 边预测实践
边预测任务，如果是预测两个节点之间是否存在边。拿到一个图数据集，我们有节点特征矩阵`x`，和哪些节点之间存在边的信息`edge_index`。`edge_index`存储的便是正样本，为了构建边预测任务，我们需要生成一些负样本，即采样一些不存在边的节点对作为负样本边，正负样本应平衡。此外要将样本分为训练集、验证集和测试集三个集合。

PyG中为我们提供了现成的方法，`train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1)`，其第一个参数为`torch_geometric.data.Data`对象，第二参数为验证集所占比例，第三个参数为测试集所占比例。该函数将自动地采样得到负样本，并将正负样本分成训练集、验证集和测试集三个集合。它用`train_pos_edge_index`、`train_neg_adj_mask`、`val_pos_edge_index`、`val_neg_edge_index`、`test_pos_edge_index`和`test_neg_edge_index`属性取代`edge_index`属性。注意`train_neg_adj_mask`与其他属性格式不同，其实该属性在后面并没有派上用场，后面我们仍然需要进行一次负样本训练集采样。

下面我们使用Cora数据集作为例子进行边预测任务说明。

1. **获取数据集并进行分析**：


```python
import os.path as osp

from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = 'Cora'
path = osp.join('dataset', dataset) # osp.dirname(osp.realpath('__file__')),
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
print(data)
ground_truth_edge_index = data.edge_index.to(device)
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
data = data.to(device)

# print(data.edge_index.shape)
# # torch.Size([2, 10556])

# for key in data.keys:
#     print(key, getattr(data, key).shape)

# x torch.Size([2708, 1433])
# val_pos_edge_index torch.Size([2, 263])
# test_pos_edge_index torch.Size([2, 527])
# train_pos_edge_index torch.Size([2, 8976])
# train_neg_adj_mask torch.Size([2708, 2708])
# val_neg_edge_index torch.Size([2, 263])
# test_neg_edge_index torch.Size([2, 527])
# 263 + 527 + 8976 = 9766 != 10556
# 263 + 527 + 8976/2 = 5278 = 10556/2
```

    Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])


观察到三个集合中正样本边的数量之和不等于原始边的数量。这是因为原始边的数量统计的是双向边的数量，在验证集正样本边和测试集正样本边中只需对一个方向的边做预测精度的衡量，对另一个方向的预测精度衡量属于重复，但在训练集还是保留双向的边

2. 构建神经网络模型


```python
import torch
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

```

用于做边预测的神经网络主要由两部分组成：其一是编码（encode），它与我们前面介绍的生成节点表征是一样的；其二是解码（decode），它边两端节点的表征生成边为真的几率（odds）。`decode_all(self, z)`用于推断（inference）阶段，我们要对输入节点所有的节点对预测存在边的几率。

3. **定义单个epoch的训练过程**


```python
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F

def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def train(data, model, optimizer):
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))

    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index).to(data.x.device)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss

```

通常在图上存在边的节点对的数量往往少于不存在边的节点对的数量。为了类平衡，在每一个`epoch`的训练过程中，我们只需要用到与正样本一样数量的负样本。综合以上两点原因，我们在每一个`epoch`的训练过程中都采样与正样本数量一样的负样本，这样我们既做到了类平衡，又增加了训练负样本的丰富性。`get_link_labels`函数用于生成完整训练集的标签。在负样本采样时，我们传递了`train_pos_edge_index`为参数，于是`negative_sampling`函数只会在训练集中不存在边的结点对中采样。

在训练阶段，我们应该只见训练集，对验证集与测试集都是不可见的，但在此阶段我们应该要完成对所有结点的编码，因此我们假设此处正样本训练集涉及到了所有的结点，这样就能实现对所有结点的编码。

4. **定义单个epoch验证与测试过程**


```python
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def test(data, model):
    model.eval()

    z = model.encode(data.x, data.train_pos_edge_index)

    results = []
    for prefix in ['val', 'test']:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return results

```

5. **运行完整的训练、验证与测试**


```python
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = 'Cora'
    path = osp.join('dataset', dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    ground_truth_edge_index = data.edge_index.to(device)
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)
    data = data.to(device)

    model = Net(dataset.num_features, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    best_val_auc = test_auc = 0
    for epoch in range(1, 101):
        loss = train(data, model, optimizer)
        val_auc, tmp_test_auc = test(data, model)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc = tmp_test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

    z = model.encode(data.x, data.train_pos_edge_index)
    final_edge_index = model.decode_all(z)
    print('ground truth edge shape:', ground_truth_edge_index.shape)
    print('final edge shape:', final_edge_index.shape)


if __name__ == "__main__":
    main()

```

    Epoch: 001, Loss: 0.6930, Val: 0.7070, Test: 0.6978
    Epoch: 002, Loss: 0.6806, Val: 0.7011, Test: 0.6978
    Epoch: 003, Loss: 0.7080, Val: 0.7039, Test: 0.6978
    Epoch: 004, Loss: 0.6752, Val: 0.7191, Test: 0.6960
    Epoch: 005, Loss: 0.6849, Val: 0.7560, Test: 0.7192
    Epoch: 006, Loss: 0.6889, Val: 0.7624, Test: 0.7277
    Epoch: 007, Loss: 0.6897, Val: 0.7480, Test: 0.7277
    Epoch: 008, Loss: 0.6886, Val: 0.7378, Test: 0.7277
    Epoch: 009, Loss: 0.6853, Val: 0.7296, Test: 0.7277
    Epoch: 010, Loss: 0.6799, Val: 0.7208, Test: 0.7277
    Epoch: 011, Loss: 0.6753, Val: 0.7134, Test: 0.7277
    Epoch: 012, Loss: 0.6774, Val: 0.7097, Test: 0.7277
    Epoch: 013, Loss: 0.6722, Val: 0.7094, Test: 0.7277
    Epoch: 014, Loss: 0.6656, Val: 0.7140, Test: 0.7277
    Epoch: 015, Loss: 0.6614, Val: 0.7189, Test: 0.7277
    Epoch: 016, Loss: 0.6570, Val: 0.7142, Test: 0.7277
    Epoch: 017, Loss: 0.6487, Val: 0.7091, Test: 0.7277
    Epoch: 018, Loss: 0.6402, Val: 0.7147, Test: 0.7277
    Epoch: 019, Loss: 0.6339, Val: 0.7331, Test: 0.7277
    Epoch: 020, Loss: 0.6237, Val: 0.7567, Test: 0.7277
    Epoch: 021, Loss: 0.6104, Val: 0.7588, Test: 0.7277
    Epoch: 022, Loss: 0.6021, Val: 0.7542, Test: 0.7277
    Epoch: 023, Loss: 0.5842, Val: 0.7479, Test: 0.7277
    Epoch: 024, Loss: 0.5713, Val: 0.7465, Test: 0.7277
    Epoch: 025, Loss: 0.5668, Val: 0.7416, Test: 0.7277
    Epoch: 026, Loss: 0.5666, Val: 0.7422, Test: 0.7277
    Epoch: 027, Loss: 0.5640, Val: 0.7452, Test: 0.7277
    Epoch: 028, Loss: 0.5666, Val: 0.7504, Test: 0.7277
    Epoch: 029, Loss: 0.5525, Val: 0.7578, Test: 0.7277
    Epoch: 030, Loss: 0.5510, Val: 0.7704, Test: 0.7654
    Epoch: 031, Loss: 0.5455, Val: 0.7838, Test: 0.7705
    Epoch: 032, Loss: 0.5364, Val: 0.7976, Test: 0.7785
    Epoch: 033, Loss: 0.5329, Val: 0.8108, Test: 0.7888
    Epoch: 034, Loss: 0.5258, Val: 0.8226, Test: 0.7978
    Epoch: 035, Loss: 0.5199, Val: 0.8334, Test: 0.8055
    Epoch: 036, Loss: 0.5144, Val: 0.8412, Test: 0.8116
    Epoch: 037, Loss: 0.5062, Val: 0.8466, Test: 0.8171
    Epoch: 038, Loss: 0.5033, Val: 0.8500, Test: 0.8201
    Epoch: 039, Loss: 0.5068, Val: 0.8528, Test: 0.8231
    Epoch: 040, Loss: 0.5042, Val: 0.8541, Test: 0.8262
    Epoch: 041, Loss: 0.5034, Val: 0.8564, Test: 0.8287
    Epoch: 042, Loss: 0.5016, Val: 0.8600, Test: 0.8316
    Epoch: 043, Loss: 0.4992, Val: 0.8642, Test: 0.8343
    Epoch: 044, Loss: 0.4960, Val: 0.8671, Test: 0.8364
    Epoch: 045, Loss: 0.4903, Val: 0.8695, Test: 0.8383
    Epoch: 046, Loss: 0.4896, Val: 0.8723, Test: 0.8400
    Epoch: 047, Loss: 0.4922, Val: 0.8757, Test: 0.8422
    Epoch: 048, Loss: 0.4848, Val: 0.8789, Test: 0.8439
    Epoch: 049, Loss: 0.4821, Val: 0.8819, Test: 0.8460
    Epoch: 050, Loss: 0.4840, Val: 0.8828, Test: 0.8482
    Epoch: 051, Loss: 0.4819, Val: 0.8844, Test: 0.8509
    Epoch: 052, Loss: 0.4840, Val: 0.8892, Test: 0.8546
    Epoch: 053, Loss: 0.4742, Val: 0.8941, Test: 0.8578
    Epoch: 054, Loss: 0.4745, Val: 0.8962, Test: 0.8597
    Epoch: 055, Loss: 0.4691, Val: 0.8967, Test: 0.8612
    Epoch: 056, Loss: 0.4733, Val: 0.8975, Test: 0.8631
    Epoch: 057, Loss: 0.4704, Val: 0.9004, Test: 0.8651
    Epoch: 058, Loss: 0.4722, Val: 0.9013, Test: 0.8664
    Epoch: 059, Loss: 0.4724, Val: 0.9026, Test: 0.8666
    Epoch: 060, Loss: 0.4670, Val: 0.9029, Test: 0.8660
    Epoch: 061, Loss: 0.4721, Val: 0.9035, Test: 0.8666
    Epoch: 062, Loss: 0.4680, Val: 0.9042, Test: 0.8682
    Epoch: 063, Loss: 0.4638, Val: 0.9041, Test: 0.8682
    Epoch: 064, Loss: 0.4647, Val: 0.9044, Test: 0.8700
    Epoch: 065, Loss: 0.4643, Val: 0.9050, Test: 0.8710
    Epoch: 066, Loss: 0.4668, Val: 0.9059, Test: 0.8726
    Epoch: 067, Loss: 0.4622, Val: 0.9068, Test: 0.8743
    Epoch: 068, Loss: 0.4586, Val: 0.9079, Test: 0.8758
    Epoch: 069, Loss: 0.4566, Val: 0.9082, Test: 0.8779
    Epoch: 070, Loss: 0.4549, Val: 0.9083, Test: 0.8804
    Epoch: 071, Loss: 0.4597, Val: 0.9099, Test: 0.8826
    Epoch: 072, Loss: 0.4533, Val: 0.9118, Test: 0.8835
    Epoch: 073, Loss: 0.4530, Val: 0.9135, Test: 0.8850
    Epoch: 074, Loss: 0.4525, Val: 0.9139, Test: 0.8859
    Epoch: 075, Loss: 0.4491, Val: 0.9148, Test: 0.8873
    Epoch: 076, Loss: 0.4546, Val: 0.9157, Test: 0.8879
    Epoch: 077, Loss: 0.4577, Val: 0.9170, Test: 0.8876
    Epoch: 078, Loss: 0.4495, Val: 0.9171, Test: 0.8875
    Epoch: 079, Loss: 0.4520, Val: 0.9173, Test: 0.8877
    Epoch: 080, Loss: 0.4514, Val: 0.9172, Test: 0.8877
    Epoch: 081, Loss: 0.4538, Val: 0.9179, Test: 0.8897
    Epoch: 082, Loss: 0.4508, Val: 0.9189, Test: 0.8907
    Epoch: 083, Loss: 0.4512, Val: 0.9193, Test: 0.8907
    Epoch: 084, Loss: 0.4501, Val: 0.9190, Test: 0.8907
    Epoch: 085, Loss: 0.4496, Val: 0.9186, Test: 0.8907
    Epoch: 086, Loss: 0.4440, Val: 0.9191, Test: 0.8907
    Epoch: 087, Loss: 0.4459, Val: 0.9200, Test: 0.8904
    Epoch: 088, Loss: 0.4494, Val: 0.9205, Test: 0.8907
    Epoch: 089, Loss: 0.4448, Val: 0.9204, Test: 0.8907
    Epoch: 090, Loss: 0.4465, Val: 0.9204, Test: 0.8907
    Epoch: 091, Loss: 0.4490, Val: 0.9205, Test: 0.8896
    Epoch: 092, Loss: 0.4454, Val: 0.9211, Test: 0.8909
    Epoch: 093, Loss: 0.4465, Val: 0.9213, Test: 0.8923
    Epoch: 094, Loss: 0.4403, Val: 0.9217, Test: 0.8925
    Epoch: 095, Loss: 0.4450, Val: 0.9224, Test: 0.8923
    Epoch: 096, Loss: 0.4390, Val: 0.9238, Test: 0.8930
    Epoch: 097, Loss: 0.4411, Val: 0.9247, Test: 0.8942
    Epoch: 098, Loss: 0.4401, Val: 0.9256, Test: 0.8955
    Epoch: 099, Loss: 0.4448, Val: 0.9254, Test: 0.8955
    Epoch: 100, Loss: 0.4384, Val: 0.9264, Test: 0.8961
    ground truth edge shape: torch.Size([2, 10556])
    final edge shape: torch.Size([2, 3311358])


## 作业

- 实践问题一：对节点预测任务，尝试用PyG中的不同的网络层去代替`GCNConv`，以及不同的层数和不同的`out_channels`。

- 实践问题二：对边预测任务，尝试用`torch_geometric.nn.Sequential`容器构造图神经网络。

- 思考问题三：如下方代码所示，我们以`data.train_pos_edge_index`为实际参数，这样采样得到的负样本可能包含验证集正样本或测试集正样本，即可能将真实的正样本标记为负样本，由此会产生冲突。但我们还是这么做，这是为什么？以及为什么在验证与测试阶段我们只根据`data.train_pos_edge_index`做结点表征的编码？

  ```python
  neg_edge_index = negative_sampling(
      edge_index=data.train_pos_edge_index,
      num_nodes=data.num_nodes,
      num_neg_samples=data.train_pos_edge_index.size(1))
  ```

## 参考文献

- `InMemoryDataset `官方文档：[`torch_geometric.data.InMemoryDataset`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset)
- `Data`官方文档：[`torch_geometric.data.Data`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data)
- 提出PubMed数据集的论文：[Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/pdf/1603.08861.pdf)
- `Planetoid`官方文档：[torch_geometric.datasets.Planetoid]([torch_geometric.datasets — pytorch_geometric 1.7.0 documentation (pytorch-geometric.readthedocs.io)](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid))
- `Sequential`官网文档：[torch_geometric.nn.Sequential](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.sequential.Sequential)
