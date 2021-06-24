# Task03 基于图神经网络的节点表征学习

## 一、基本知识

* 基于图神经网络做预测：首先需要生成节点表征。高质量节点表征应该能用于衡量节点的相似性，然后基于节点表征可以实现高准确性的节点预测或边预测，因此节点表征的生成是图节点预测和边预测任务成功的关键

* 节点预测任务基本过程：我们拥有一个图，图上有很多节点，部分节点的标签已知，剩余节点的标签未知。将节点的属性（`x`）、边的端点信息（`edge_index`）、边的属性（`edge_attr`，如果有的话）输入到多层图神经网络，经过图神经网络每一层的一次节点间信息传递，图神经网络为节点生成节点表征。
* 目标：根据节点的属性(可以是类别型、也可以是数值型)、边的信息、边的属性（如果有的话）、已知的节点预测标签，对未知标签的节点做预测。


```python
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('======================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

```


    Dataset: Cora():
    ======================
    Number of graphs: 1
    Number of features: 1433
    Number of classes: 7
    
    Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
    ======================
    Number of nodes: 2708
    Number of edges: 10556
    Average node degree: 3.90
    Number of training nodes: 140
    Training node label rate: 0.05
    Contains isolated nodes: False
    Contains self-loops: False
    Is undirected: True


## 二、比较多层图神经网络与MLP

### 2.1 准备工作

### 2.1.1 获取并分析数据集


```python
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
# Planetoid.url ='https://gitee.com/rongqinchen/planetoid/tree/master/data'
# 数据转换在将数据输入到神经网络之前修改数据，这一功能可用于实现数据规范化或数据增强
# 使用NormalizeFeatures：进行节点特征归一化，使各节点特征总和为1
dataset = Planetoid(root='./dataset/Cora', name='Cora', transform=NormalizeFeatures())

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('======================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
```

    Dataset: Cora():
    ======================
    Number of graphs: 1
    Number of features: 1433
    Number of classes: 7
    
    Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
    ======================
    Number of nodes: 2708
    Number of edges: 10556
    Average node degree: 3.90
    Number of training nodes: 140
    Training node label rate: 0.05
    Contains isolated nodes: False
    Contains self-loops: False
    Is undirected: True


结果解读：
* Cora图拥有2708个节点和10,556条边，平均节点度为3.9
* 仅使用140个有真实标签的节点（每类20个）用于训练
* 有标签的节点的比例只占到5%
* 有向无向：无向图，不存在孤立节点（每个文档至少有一个引文）

### 2.1.2 可视化节点表征分布的方法

* manifold:流形数据，像绳结一样的数据，虽然在高维空间中可分，但是在人眼所看到的低维空间中，绳结中的绳子是互相重叠的不可分的。
* TSNE:数据降维与可视化
1. 将数据点之间的相似度转换为概率，主要关注数据的局部结构
* 使用t-SNE的缺点：
1. t-SNE的计算复杂度很高，在数百万个样本数据集中可能需要几个小时，而PCA可以在几秒钟或几分钟内完成
2. Barnes-Hut t-SNE方法（下面讲）限于二维或三维嵌入。
3. 算法是随机的，具有不同种子的多次实验可以产生不同的结果。虽然选择loss最小的结果就行，但可能需要多次实验以选择超参数。
4. 全局结构未明确保留。这个问题可以通过PCA初始化点（使用init ='pca'）来缓解。


```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(h, color):
    # n_components:嵌入空间的维度（结果空间的维度）
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    
    plt.scatter(z[:, 0], z[:,1], s=70, c=color, cmap='set2')
    plt.show()
```

### 2.2 MLP应用于图节点分类
MLP网络结构：由两个全连接层，第一个全连接层后加ReLU增加非线性表达能力，并且进行dropout防止过拟合。


```python
import torch
from torch.nn import Linear
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(MLP, self).__init__()
        torch.manual_seed(2021)
        self.fc1 = Linear(dataset.num_features, hidden_channels)
        self.fc2 = Linear(hidden_channels, dataset.num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

model = MLP(hidden_channels=16)
print(model)
```

    MLP(
      (fc1): Linear(in_features=1433, out_features=16, bias=True)
      (fc2): Linear(in_features=16, out_features=7, bias=True)
    )


训练MLP网络
* 损失函数：交叉熵损失函数
* 优化器：Adam


```python
model = MLP(hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad() # Clear gradients
    out = model(data.x)
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) # compute the loss solely based on the training nodes
    loss.backward()
    optimizer.step()
    return loss

for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
```

    Epoch: 001, Loss: 1.9527
    Epoch: 002, Loss: 1.9491
    Epoch: 003, Loss: 1.9459
    Epoch: 004, Loss: 1.9414
    Epoch: 005, Loss: 1.9348
    Epoch: 006, Loss: 1.9288
    Epoch: 007, Loss: 1.9198
    Epoch: 008, Loss: 1.9136
    Epoch: 009, Loss: 1.9111
    Epoch: 010, Loss: 1.9031
    Epoch: 020, Loss: 1.7829
    Epoch: 050, Loss: 1.1777
    Epoch: 090, Loss: 0.6540
    Epoch: 120, Loss: 0.5510
    Epoch: 150, Loss: 0.4577
    Epoch: 180, Loss: 0.4398
    Epoch: 200, Loss: 0.2876


测试MLP模型


```python
def test():
    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1) # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask] 
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) 
    return test_acc

test_acc = test()
print(f'Test Accuracy: {test_acc: .4f}')
```

    Test Accuracy:  0.5850


### 2.3 GCN应用于图节点分类

#### GCN定义
GCN 神经网络层来源于论文“[Semi-supervised Classification with Graph Convolutional Network](https://arxiv.org/abs/1609.02907)”，其数学定义为，
$$
\mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
\mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
$$
其中$\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}$表示插入自环的邻接矩阵，$\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}$表示其对角线度矩阵。邻接矩阵可以包括不为$1$的值，当邻接矩阵不为`{0,1}`值时，表示邻接矩阵存储的是边的权重。$\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
\mathbf{\hat{D}}^{-1/2}$对称归一化矩阵。

它的节点式表述为：
$$
\mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup
\{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j
$$
其中，$\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}$，$e_{j,i}$表示从源节点$j$到目标节点$i$的边的对称归一化系数（默认值为1.0）。

#### PyG 中 `GCNConv` 模块说明
`GCNConv`构造函数接口：

```python
GCNConv(in_channels: int, out_channels: int, improved: bool = False, cached: bool = False, add_self_loops: bool = True, normalize: bool = True, bias: bool = True, **kwargs)
```

其中：

- `in_channels `：输入数据维度；
- `out_channels `：输出数据维度；
- `improved `：如果为`true`，$\mathbf{\hat{A}} = \mathbf{A} + 2\mathbf{I}$，其目的在于增强中心节点自身信息；
- `cached `：是否存储$\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}$的计算结果以便后续使用，这个参数只应在归纳学习（transductive learning）的景中设置为`true`；
- `add_self_loops `：是否在邻接矩阵中增加自环边；
- `normalize `：是否添加自环边并在运行中计算对称归一化系数；
- `bias `：是否包含偏置项。

详细内容参阅[GCNConv官方文档](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)。

#### 基于GCN图神经网络进行图节点分类
通过将`torch.nn.Linear` layers 替换为PyG的GNN Conv Layers，可以将MLP模型转化为GNN模型

```python
from torch_geometric.nn import GCNConv
# import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
# plt.style.use("ggplot")
from sklearn.manifold import TSNE

def visualize(h, color):
    # n_components:嵌入空间的维度（结果空间的维度）
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    
#     cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap='rainbow')# "Blues_r"
    plt.show()

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=16)
print(model)

## 可视化未训练的GCN网络的节点表征---7维特征的节点被嵌入到2维平面上
model = GCN(hidden_channels=16)
model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y)
```

    GCN(
      (conv1): GCNConv(1433, 16)
      (conv2): GCNConv(16, 7)
    )


![](https://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/output_18_1.png)    



```python
## 训练GCN网络
model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
```

    Epoch: 001, Loss: 1.9451
    Epoch: 002, Loss: 1.9384
    Epoch: 003, Loss: 1.9307
    Epoch: 004, Loss: 1.9227
    Epoch: 005, Loss: 1.9126
    Epoch: 006, Loss: 1.9076
    Epoch: 007, Loss: 1.8917
    Epoch: 008, Loss: 1.8809
    Epoch: 009, Loss: 1.8728
    Epoch: 010, Loss: 1.8616
    Epoch: 020, Loss: 1.7184
    Epoch: 050, Loss: 1.1714
    Epoch: 090, Loss: 0.6769
    Epoch: 120, Loss: 0.5064
    Epoch: 150, Loss: 0.4119
    Epoch: 180, Loss: 0.3186
    Epoch: 200, Loss: 0.3006



```python
## 测试GCN网络
def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
```

    Test Accuracy: 0.8140



```python
## 可视化经过GCN网络分类的数据
model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y)
```


​    
![png](https://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/output_21_0.png)
​    


结果分析：通过简单地将线性层替换成GCN层，可以达到81.4%的测试准确率！与前面的仅获得59%的测试准确率的MLP分类器相比，现在的分类器准确性要高得多。这表明节点的邻接信息在取得更好的准确率方面起着关键作用。

最后还可以通过可视化训练过的模型输出的节点表征来再次验证这一点，现在同类节点的聚集在一起的情况更加明显了。

### 2.4 GAT应用于图节点分类

#### GAT定义
图注意网络（GAT）来源于论文 [Graph Attention Networks](https://arxiv.org/abs/1710.10903)。其数学定义为，
$$
\mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
\sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
$$
其中注意力系数$\alpha_{i,j}$的计算方法为，
$$
\alpha_{i,j} =
\frac{
\exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
[\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
\right)\right)}
{\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
\exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
[\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
\right)\right)}.
$$

#### PyG中`GATConv` 模块说明

`GATConv`构造函数接口：

```python
GATConv(in_channels: Union[int, Tuple[int, int]], out_channels: int, heads: int = 1, concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.0, add_self_loops: bool = True, bias: bool = True, **kwargs)
```

其中：

- `in_channels `：输入数据维度；
- `out_channels `：输出数据维度；
- `heads `：在`GATConv`使用多少个注意力模型（Number of multi-head-attentions）；
- `concat `：如为`true`，不同注意力模型得到的节点表征被拼接到一起（表征维度翻倍），否则对不同注意力模型得到的节点表征求均值；

详细内容请参阅[GATConv官方文档](

#### 基于GAT图神经网络进行图节点分类


```python
import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(dataset.num_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```


```python
## 训练GAT网络
model = GAT(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
```

    Epoch: 001, Loss: 1.9457
    Epoch: 002, Loss: 1.9400
    Epoch: 003, Loss: 1.9350
    Epoch: 004, Loss: 1.9267
    Epoch: 005, Loss: 1.9173
    Epoch: 006, Loss: 1.9071
    Epoch: 007, Loss: 1.8998
    Epoch: 008, Loss: 1.8866
    Epoch: 009, Loss: 1.8834
    Epoch: 010, Loss: 1.8757
    Epoch: 020, Loss: 1.7236
    Epoch: 050, Loss: 0.9993
    Epoch: 090, Loss: 0.4305
    Epoch: 120, Loss: 0.3186
    Epoch: 150, Loss: 0.2666
    Epoch: 180, Loss: 0.1818
    Epoch: 200, Loss: 0.1926



```python
## 测试GCN网络
def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
```

    Test Accuracy: 0.7380



```python
## 可视化经过GAT网络分类的数据
model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y)
```


​    
![png](https://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/output_29_0.png)
​    


## 三、对比结果分析
在节点表征学习任务中，MLP,GCN和GAT三个网络的对比结果如下：
* MLP只考虑了节点自身属性，**忽略了节点之间的连接关系**，它的结果是最差
* GCN与GAT节点分类器，**同时考虑了节点自身属性与周围邻居节点的属性**，它们的结果优于MLP节点分类器。从中可以看出**邻居节点的信息对于节点分类任务的重要性**

**基于图神经网络的节点表征的学习遵循消息传递范式**：

- 在邻居节点信息变换阶段，GCN与GAT都对邻居节点做归一化和线性变换（两个操作不分前后）；
- 在邻居节点信息聚合阶段都将变换后的邻居节点信息做求和聚合；
- 在中心节点信息变换阶段只是简单返回邻居节点信息聚合阶段的聚合结果。

GCN与GAT的区别在于邻居节点信息聚合过程中的**归一化方法不同**：

- 前者根据中心节点与邻居节点的度计算归一化系数，后者根据中心节点与邻居节点的相似度计算归一化系数。
- 前者的归一化方式依赖于图的拓扑结构，不同节点其自身的度不同、其邻居的度也不同，在一些应用中可能会影响泛化能力。
- 后者的归一化方式依赖于中心节点与邻居节点的相似度，相似度是训练得到的，因此不受图的拓扑结构的影响，在不同的任务中都会有较好的泛化表现。

## 作业：
此篇文章涉及的代码可见于`codes/learn_node_representation.ipynb`，请参照这份代码使用PyG中不同的图卷积层在PyG的不同数据上实现节点分类或回归任务


```python
import torch
## 配置环境
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

## 下载数据集
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
# Planetoid.url ='https://gitee.com/rongqinchen/planetoid/tree/master/data'
# 数据转换在将数据输入到神经网络之前修改数据，这一功能可用于实现数据规范化或数据增强
# 使用NormalizeFeatures：进行节点特征归一化，使各节点特征总和为1
dataset = Planetoid(root='./dataset/Citeseer', name='Citeseer', transform=NormalizeFeatures())
# dataset_cora = Planetoid(root='./dataset/Citeseer', name='citeseer')

# 提取data，并转换为device格式
data = dataset[0].to(device)
# 打印dataset的属性
print(dataset.num_classes)  # 标签的类别数量
print(dataset.num_node_features)  # 节点特征的维度
print(len(dataset))  # 数据集图的个数
# 打印data
print(data)
```

    cpu
    6
    3703
    1
    Data(edge_index=[2, 9104], test_mask=[3327], train_mask=[3327], val_mask=[3327], x=[3327, 3703], y=[3327])



```python
## model

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
%matplotlib inline
# plt.style.use("ggplot")
from sklearn.manifold import TSNE

## MLP
class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(MLP, self).__init__()
        torch.manual_seed(2021)
        self.fc1 = Linear(dataset.num_features, hidden_channels)
        self.fc2 = Linear(hidden_channels, dataset.num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

model1 = MLP(hidden_channels=16)
print(model1)

def visualize(h, color):
    # n_components:嵌入空间的维度（结果空间的维度）
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    
#     cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap='rainbow')# "Blues_r"
    plt.show()

## GCN
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model2 = GCN(hidden_channels=16)
print(model2)

## GAT
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(dataset.num_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
model3 = GCN(hidden_channels=16)
print(model3)
```

    MLP(
      (fc1): Linear(in_features=3703, out_features=16, bias=True)
      (fc2): Linear(in_features=16, out_features=6, bias=True)
    )
    GCN(
      (conv1): GCNConv(3703, 16)
      (conv2): GCNConv(16, 6)
    )
    GCN(
      (conv1): GCNConv(3703, 16)
      (conv2): GCNConv(16, 6)
    )



```python
## train MLP
model1 = MLP(hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model1.train()
    optimizer.zero_grad() # Clear gradients
    out = model1(data.x)
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) # compute the loss solely based on the training nodes
    loss.backward()
    optimizer.step()
    return loss

for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
```

    Epoch: 001, Loss: 1.8000
    Epoch: 002, Loss: 1.7956
    Epoch: 003, Loss: 1.7915
    Epoch: 004, Loss: 1.7863
    Epoch: 005, Loss: 1.7808
    Epoch: 006, Loss: 1.7730
    Epoch: 007, Loss: 1.7644
    Epoch: 008, Loss: 1.7539
    Epoch: 009, Loss: 1.7485
    Epoch: 010, Loss: 1.7412
    Epoch: 020, Loss: 1.6199
    Epoch: 050, Loss: 0.9766
    Epoch: 080, Loss: 0.6773
    Epoch: 100, Loss: 0.5956
    Epoch: 120, Loss: 0.5405
    Epoch: 150, Loss: 0.5041
    Epoch: 180, Loss: 0.5316
    Epoch: 200, Loss: 0.3485



```python
## train GCN网络
model2 = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model2.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model2.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model2(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
```

    Epoch: 001, Loss: 1.7920
    Epoch: 002, Loss: 1.7877
    Epoch: 003, Loss: 1.7849
    Epoch: 004, Loss: 1.7775
    Epoch: 005, Loss: 1.7702
    Epoch: 006, Loss: 1.7665
    Epoch: 007, Loss: 1.7579
    Epoch: 008, Loss: 1.7495
    Epoch: 009, Loss: 1.7433
    Epoch: 010, Loss: 1.7346
    Epoch: 020, Loss: 1.6497
    Epoch: 050, Loss: 1.2842
    Epoch: 080, Loss: 0.9321
    Epoch: 120, Loss: 0.7004
    Epoch: 150, Loss: 0.5929
    Epoch: 180, Loss: 0.5230
    Epoch: 200, Loss: 0.4934



```python
## train GAT
model3 = GAT(hidden_channels=16)
optimizer = torch.optim.Adam(model3.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model3.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model3(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
```

    Epoch: 001, Loss: 1.7918
    Epoch: 002, Loss: 1.7879
    Epoch: 003, Loss: 1.7821
    Epoch: 004, Loss: 1.7770
    Epoch: 005, Loss: 1.7697
    Epoch: 006, Loss: 1.7670
    Epoch: 007, Loss: 1.7562
    Epoch: 008, Loss: 1.7546
    Epoch: 009, Loss: 1.7427
    Epoch: 010, Loss: 1.7339
    Epoch: 020, Loss: 1.6383
    Epoch: 050, Loss: 1.1420
    Epoch: 080, Loss: 0.6670
    Epoch: 120, Loss: 0.4451
    Epoch: 150, Loss: 0.3148
    Epoch: 180, Loss: 0.2881
    Epoch: 200, Loss: 0.2010



```python
def test():
    model1.eval()
    out1 = model1(data.x)
    pred = out1.argmax(dim=1) # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask] 
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) 
    return test_acc

test_acc = test()
print(f'Test Accuracy: {test_acc: .4f}')
```

    Test Accuracy:  0.5640



```python
## 测试GCN网络
def test():
      model2.eval()
      out2 = model2(data.x, data.edge_index)
      pred = out2.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
```

    Test Accuracy: 0.7120



```python
## 测试GCN网络
def test():
      model3.eval()
      out3 = model3(data.x, data.edge_index)
      pred = out3.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
```

    Test Accuracy: 0.6100


# 补充--解决dataset下载存在的问题
* 问题描述：无法下载数据集
* 问题分析：需要科学上网，从github下载数据集；所用的服务器宕机后没有设置连接外网！
* 解决方案：

   方法一：不直接从github下载，先将文件下载到本地
        1. 在微云（https://share.weiyun.com/ccvyvTfi） 下载数据集，并解压得到data数据集，将其上传到服务器，服务器的数据集路径为：‘./dataset/Cora/Cora/raw’
        2. 更改Planetoid 函数中的root路径为‘./dataset/Cora’即可 
   
   方法二：将服务器设置成能连通外网的模式，然后再科学上网，直接运行案例代码！

# 参考链接
- TSNE：https://www.deeplearn.me/2137.html
- PyG中内置的数据转换方法：[torch-geometric-transforms](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch-geometric-transforms)
- 一个可视化高纬数据的工具：[t-distributed Stochastic Neighbor Embedding](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- 提出GCN的论文：[Semi-supervised Classification with Graph Convolutional Network](https://arxiv.org/abs/1609.02907)
- GCNConv官方文档：[torch_geometric.nn.conv.GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)
- 提出GAT的论文： [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
