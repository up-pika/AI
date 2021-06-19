# Task02 消息传递图神经网络

## 一、 消息传递范式介绍

* **消息传递范式是一种聚合邻接节点信息来更新中心节点信息的范式--图神经网络生成节点表征的范式**，它将卷积算子推广到了不规则数据领域，实现了图与神经网络的连接。**该范式包含这样三个步骤：(1)邻接节点信息变换、(2)邻接节点信息聚合到中心节点、(3)聚合信息变换**。`MessagePassing`基类可以封装“消息传递”的运行流程。在PyG中，`MessagePassing`基类是所有基于消息传递范式的图神经网络的基类，它大大地方便了我们对图神经网络的构建。
* 神经网络的生成节点表征的操作也称为节点嵌入

## 二、`MessagePassing`基类初步分析

* 用途：封装“消息传递”的运行流程

* 构造一个最简单的消息传递图神经网络类，只需定义**[`message()`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.message_passing.MessagePassing.message)方法（$\phi$）**、**[`update()`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.message_passing.MessagePassing.update)方法（$\gamma$）**，以及使用的**消息聚合方案**（`aggr="add"`、`aggr="mean"`或`aggr="max"`）。这一切由以下方法共同作用而成：

  - `MessagePassing(aggr="add", flow="source_to_target", node_dim=-2)`（对象初始化方法）： 
    - `aggr`：定义要使用的聚合方案（"add"、"mean "或 "max"）；
    - `flow`：定义消息传递的流向（"source_to_target "或 "target_to_source"）；
    - `node_dim`：定义沿着哪个维度传播，默认值为`-2`，也就是节点表征张量（Tensor）的哪一个维度是节点维度。节点表征张量`x`形状为`[num_nodes, num_features]`，其第0维度（也是第-2维度）是节点维度，其第1维度（也是第-1维度）是节点表征维度，所以我们可以设置`node_dim=-2`。
    - 注：`MessagePassing(……)`等同于`MessagePassing.__init__(……)`
  - `MessagePassing.propagate(edge_index, size=None, **kwargs)`：
    - 开始传递消息的起始调用，在此方法中`message`、`update`等方法被调用。
    - 它以`edge_index`（边的端点的索引）和`flow`（消息的流向）以及一些额外的数据为参数。
    - 请注意，`propagate()`不仅限于基于形状为`[N, N]`的对称邻接矩阵进行“消息传递过程”。基于非对称的邻接矩阵进行消息传递（当图为二部图时），需要传递参数`size=(N, M)`。
    - 如果设置`size=None`，则认为邻接矩阵是对称的。
  - `MessagePassing.message(...)`：
    - 首先确定要给节点$i$传递消息的边的集合：
      - 如果`flow="source_to_target"`，则是$(j,i) \in \mathcal{E}$的边的集合；
      - 如果`flow="target_to_source"`，则是$(i,j) \in \mathcal{E}$的边的集合。
    - 接着为各条边创建要传递给节点$i$的消息，即实现$\phi$函数。
    - `MessagePassing.message(...)`方法可以接收传递给`MessagePassing.propagate(edge_index, size=None, **kwargs)`方法的所有参数，我们在`message()`方法的参数列表里定义要接收的参数，例如我们要接收`x,y,z`参数，则我们应定义`message(x,y,z)`方法。
    - 传递给`propagate()`方法的参数，如果是节点的属性的话，可以被拆分成属于中心节点的部分和属于邻接节点的部分，只需在变量名后面加上`_i`或`_j`。例如，我们自己定义的`meassage`方法包含参数`x_i`，那么首先`propagate()`方法将节点表征拆分成中心节点表征和邻接节点表征，接着`propagate()`方法调用`message`方法并传递中心节点表征给参数`x_i`。而如果我们自己定义的`meassage`方法包含参数`x_j`，那么`propagate()`方法会传递邻接节点表征给参数`x_j`。
    - 我们用$i$表示“消息传递”中的中心节点，用$j$表示“消息传递”中的邻接节点。
  - `MessagePassing.aggregate(...)`：
    - 将从源节点传递过来的消息聚合在目标节点上，一般可选的聚合方式有`sum`, `mean`和`max`。
  - `MessagePassing.message_and_aggregate(...)`：
    - 在一些场景里，邻接节点信息变换和邻接节点信息聚合这两项操作可以融合在一起，那么我们可以在此方法里定义这两项操作，从而让程序运行更加高效。
  - `MessagePassing.update(aggr_out, ...)`: 
    - 为每个节点$i \in \mathcal{V}$更新节点表征，即实现$\gamma$函数。此方法以`aggregate`方法的输出为第一个参数，并接收所有传递给`propagate()`方法的参数。

  以上内容来源于[The “MessagePassing” Base Class](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#the-messagepassing-base-class)。

## 三、`MessagePassing`子类实例

以继承`MessagePassing`基类的`GCNConv`类为例，实现一个简单的图神经网络。

**[`GCNConv`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)的数学定义为**
$$
\mathbf{x}_i^{(k)} = \sum_{j \in \mathcal{N}(i) \cup \{ i \}} \frac{1}{\sqrt{\deg(i)} \cdot \sqrt{\deg(j)}} \cdot \left( \mathbf{\Theta} \cdot \mathbf{x}_j^{(k-1)} \right),
$$
其中，邻接节点的表征$\mathbf{x}_j^{(k-1)}$首先通过与权重矩阵$\mathbf{\Theta}$相乘进行变换，然后按端点的度$\deg(i), \deg(j)$进行归一化处理，最后进行求和。这个公式可以分为以下几个步骤：

1. 向邻接矩阵添加自环边。
1. 对节点表征做线性转换。
1. 计算归一化系数。
1. 归一化邻接节点的节点表征。
1. 将相邻节点表征相加（"求和 "聚合）。

步骤1-3通常是在消息传递发生之前计算的。步骤4-5可以使用[`MessagePassing`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.message_passing.MessagePassing)基类轻松处理。该层的全部实现如下所示。

```python
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add', flow='source_to_target')
        # "Add" aggregation (Step 5).
        # flow='source_to_target' 表示消息从源节点传播到目标节点
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
```

`GCNConv`继承了`MessagePassing`并以"求和"作为领域节点信息聚合方式。该层的所有逻辑都发生在其`forward()`方法中。

在这里，首先使用`torch_geometric.utils.add_self_loops()`函数向我们的边索引添加自循环边（步骤1），以及通过调用`torch.nn.Linear`实例对节点表征进行线性变换（步骤2）。`propagate()`方法也在`forward`方法中被调用，`propagate()`方法被调用后节点间的信息传递开始执行。

归一化系数是由每个节点的节点度得出的，它被转换为每条边的节点度。结果被保存在形状为`[num_edges,]`的变量`norm`中（步骤3）。

在[`message()`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.message_passing.MessagePassing.message)方法中，需要通过`norm`对邻接节点表征`x_j`进行归一化处理。

通过以上内容的学习，让我掌握了**创建一个仅包含一次“消息传递过程”的图神经网络的方法**。如下方代码所示，可以方便地初始化和调用它：

```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='dataset/Cora', name='Cora')
data = dataset[0]

net = GCNConv(data.num_features, 64)
h_nodes = net(data.x, data.edge_index)
print(h_nodes.shape)
```

**通过串联多个这样的简单图神经网络，我们就可以构造复杂的图神经网络模型**。我们将在[第5节](5-基于图神经网络的节点表征学习.md)介绍复杂图神经网络模型的构建。

以上主要内容来源于[Implementing the GCN Layer](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#implementing-the-gcn-layer)。

## 四、`MessagePassing`基类剖析

在`__init__()`方法中，我们看到程序会检查子类是否实现了`message_and_aggregate()`方法，并将检查结果赋值给`fuse`属性。

```python
class MessagePassing(torch.nn.Module):
	def __init__(self, aggr: Optional[str] = "add", flow: str = "source_to_target", node_dim: int = -2):
        super(MessagePassing, self).__init__()
		# 此处省略n行代码
        # Support for "fused" message passing.
        self.fuse = self.inspector.implements('message_and_aggregate')
		# 此处省略n行代码

```

“消息传递过程”是从`propagate`方法被调用开始执行的。

```python
class MessagePassing(torch.nn.Module):
    # 此处省略n行代码
    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
    	# 此处省略n行代码
        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index, size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute('message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)
        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)
    		# 此处省略n行代码
            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

```

参数简介：

- `edge_index`: 边端点索引，它可以是`Tensor`类型或`SparseTensor`类型。 
  - 当flow="source_to_target"时，节点`edge_index[0]`的信息将被传递到节点`edge_index[1]`，
  - 当flow="target_to_source"时，节点`edge_index[1]`的信息将被传递到节点`edge_index[0]`
- `size`: 邻接节点的数量与中心节点的数量。
  - 对于普通图，邻接节点的数量与中心节点的数量都是N，我们可以不给size传参数，即让size取值为默认值None。
  - 对于二部图，邻接节点的数量与中心节点的数量分别记为M, N，于是我们需要给size参数传一个元组`(M, N)`。
- `kwargs`: 图其他属性或额外的数据。

`propagate()`方法首先检查`edge_index`是否为`SparseTensor`类型以及是否子类实现了`message_and_aggregate()`方法，如是就执行子类的`message_and_aggregate`方法；否则依次执行子类的`message(),aggregate(),update()`三个方法。

## 五、`message`方法的覆写

前面我们介绍了，传递给`propagate()`方法的参数，如果是节点的属性的话，可以被拆分成属于中心节点的部分和属于邻接节点的部分，只需在变量名后面加上`_i`或`_j`。现在我们有一个额外的节点属性，节点的度`deg`，我们希望`meassge`方法还能接收中心节点的度，我们对前面`GCNConv`的`message`方法进行改造得到新的`GCNConv`类：

```python
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add', flow='source_to_target')
        # "Add" aggregation (Step 5).
        # flow='source_to_target' 表示消息从源节点传播到目标节点
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm, deg=deg.view((-1, 1)))

    def message(self, x_j, norm, deg_i):
        # x_j has shape [E, out_channels]
        # deg_i has shape [E, 1]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j * deg_i


from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='dataset/Cora', name='Cora')
data = dataset[0]

net = GCNConv(data.num_features, 64)
h_nodes = net(data.x, data.edge_index)
print(h_nodes.shape)

```

若一个数据可以被拆分成属于中心节点的部分和属于邻接节点的部分，其形状必须是`[num_nodes, *]`，因此在上方代码的第`29`行，我们执行了`deg.view((-1, 1))`操作，使得数据形状为`[num_nodes, 1]`，然后才将数据传给`propagate()`方法。

## 六、`aggregate`方法的覆写

在前面的例子的基础上，我们增加如下的`aggregate`方法。通过观察运行结果我们可以看到，我们覆写的`aggregate`方法被调用，同时在`super(GCNConv, self).__init__(aggr='add')`中传递给`aggr`参数的值被存储到了`self.aggr`属性中。

```python
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add', flow='source_to_target')
        # "Add" aggregation (Step 5).
        # flow='source_to_target' 表示消息从源节点传播到目标节点
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm, deg=deg.view((-1, 1)))

    def message(self, x_j, norm, deg_i):
        # x_j has shape [E, out_channels]
        # deg_i has shape [E, 1]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j * deg_i

    def aggregate(self, inputs, index, ptr, dim_size):
        print('self.aggr:', self.aggr)
        print("`aggregate` is called")
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
        

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='dataset/Cora', name='Cora')
data = dataset[0]

net = GCNConv(data.num_features, 64)
h_nodes = net(data.x, data.edge_index)
print(h_nodes.shape)

```

## 七、`message_and_aggregate`方法的覆写

在一些案例中，“消息传递”与“消息聚合”可以融合在一起。对于这种情况，我们可以覆写`message_and_aggregate`方法，在`message_and_aggregate`方法中一块实现“消息传递”与“消息聚合”，这样能使程序的运行更加高效。

```python
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_sparse import SparseTensor

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add', flow='source_to_target')
        # "Add" aggregation (Step 5).
        # flow='source_to_target' 表示消息从源节点传播到目标节点
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        adjmat = SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.shape[1]))
        # 此处传的不再是edge_idex，而是SparseTensor类型的Adjancency Matrix
        return self.propagate(adjmat, x=x, norm=norm, deg=deg.view((-1, 1)))

    def message(self, x_j, norm, deg_i):
        # x_j has shape [E, out_channels]
        # deg_i has shape [E, 1]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j * deg_i

    def aggregate(self, inputs, index, ptr, dim_size):
        print('self.aggr:', self.aggr)
        print("`aggregate` is called")
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

    def message_and_aggregate(self, adj_t, x, norm):
        print('`message_and_aggregate` is called')
        # 没有实现真实的消息传递与消息聚合的操作
 
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='dataset/Cora', name='Cora')
data = dataset[0]

net = GCNConv(data.num_features, 64)
h_nodes = net(data.x, data.edge_index)
# print(h_nodes.shape)

```

运行程序后我们可以看到，虽然我们同时覆写了`message`方法和`aggregate`方法，然而只有`message_and_aggregate`方法被执行。

## 八、覆写`update`方法

```python
from torch_geometric.datasets import Planetoid
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_sparse import SparseTensor


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add', flow='source_to_target')
        # "Add" aggregation (Step 5).
        # flow='source_to_target' 表示消息从源节点传播到目标节点
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        adjmat = SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.shape[1]))
        # 此处传的不再是edge_idex，而是SparseTensor类型的Adjancency Matrix
        return self.propagate(adjmat, x=x, norm=norm, deg=deg.view((-1, 1)))

    def message(self, x_j, norm, deg_i):
        # x_j has shape [E, out_channels]
        # deg_i has shape [E, 1]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j * deg_i

    def aggregate(self, inputs, index, ptr, dim_size):
        print('self.aggr:', self.aggr)
        print("`aggregate` is called")
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

    def message_and_aggregate(self, adj_t, x, norm):
        print('`message_and_aggregate` is called')
        # 没有实现真实的消息传递与消息聚合的操作

    def update(self, inputs, deg):
        print(deg)
        return inputs


dataset = Planetoid(root='dataset/Cora', name='Cora')
data = dataset[0]

net = GCNConv(data.num_features, 64)
h_nodes = net(data.x, data.edge_index)
# print(h_nodes.shape)

```

`update`方法接收聚合的输出作为第一个参数，此外还可以接收传递给`propagate`方法的任何参数。在上方的代码中，我们覆写的`update`方法接收了聚合的输出作为第一个参数，此外接收了传递给`propagate`的`deg`参数。

## 九、作业

1. 请总结`MessagePassing`基类的运行流程。

   Message Passing 根据上面讨论的的框架公式，在设计Message Passing 的流程可以归纳为以下几点:

   1. 定义和选取 message 函数( 𝜙() )，并根据图的节点信息的输入$(x^{k−1}_i,x^{k−1}_j,e_{i,j}) $对输入进行变换
   2. 定义和选取 aggregate 函数 , 将从源节点传递过来的消息聚合在目标节点上，对转换后的信息进行邻居节点的信息聚合处理， 常用的有sum, mean, max之类的
   3. 定义和选取update()函数（ 𝛾() ），把原本的节点信息 和 聚合后邻居节点信息函数输出的信息进行整合，更新当前的节点信息。

2. 请复现一个一层的图神经网络的构造，总结通过继承`MessagePassing`基类来构造自己的图神经网络类的规范。

   所设计的单层图神经网络公式：
   $$
   \mathbf{x}_i^{(k)} = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \{ i \}} \frac{1}{\sqrt{\deg(i)} \cdot \sqrt{\deg(j)}} \cdot \left( \mathbf{\Theta} \cdot \mathbf{x}_j^{(k-1)} \right)\right)+ \mathbf{\Theta} \cdot \mathbf{x}_j^{(k-1)}
   $$

   ```
   from torch_geometric.datasets import Planetoid
   import torch
   from torch import nn, Tensor
   from torch_geometric.nn import MessagePassing
   from torch_geometric.utils import add_self_loops, degree
   from torch_sparse import SparseTensor, matmul
   
   
   class GCNConv(MessagePassing):
       def __init__(self, in_channels, out_channels):
           super(GCNConv, self).__init__(aggr='mean', flow='source_to_target')
           # "Add" aggregation (Step 5).
           # flow='source_to_target' 表示消息从源节点传播到目标节点
           self.lin = torch.nn.Linear(in_channels, out_channels)
           self.relu = torch.nn.ReLU()
   
           
       def propagate(self, edge_index, size=None, **kwargs):
           # I just copy the source copy from PyG website
           r"""The initial call to start propagating messages.
   
           Args:
               edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                   :obj:`torch_sparse.SparseTensor` that defines the underlying
                   graph connectivity/message passing flow.
                   :obj:`edge_index` holds the indices of a general (sparse)
                   assignment matrix of shape :obj:`[N, M]`.
                   If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                   shape must be defined as :obj:`[2, num_messages]`, where
                   messages from nodes in :obj:`edge_index[0]` are sent to
                   nodes in :obj:`edge_index[1]`
                   (in case :obj:`flow="source_to_target"`).
                   If :obj:`edge_index` is of type
                   :obj:`torch_sparse.SparseTensor`, its sparse indices
                   :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                   and :obj:`col = edge_index[0]`.
                   The major difference between both formats is that we need to
                   input the *transposed* sparse adjacency matrix into
                   :func:`propagate`.
               size (tuple, optional): The size :obj:`(N, M)` of the assignment
                   matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                   If set to :obj:`None`, the size will be automatically inferred
                   and assumed to be quadratic.
                   This argument is ignored in case :obj:`edge_index` is a
                   :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
               **kwargs: Any additional data which is needed to construct and
                   aggregate messages, and to update node embeddings.
           """
           size = self.__check_input__(edge_index, size)
   
           # Run "fused" message and aggregation (if applicable).
           if (isinstance(edge_index, SparseTensor) and self.fuse
                   and not self.__explain__):
               coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                            size, kwargs)
               print("Using self-defined message-passing")
               msg_aggr_kwargs = self.inspector.distribute(
                   'message_and_aggregate', coll_dict)
               out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
   
               update_kwargs = self.inspector.distribute('update', coll_dict)
               return self.update(out, **update_kwargs)
   
           # Otherwise, run both functions in separation.
           elif isinstance(edge_index, Tensor) or not self.fuse:
               coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                            kwargs)
   
               msg_kwargs = self.inspector.distribute('message', coll_dict)
               out = self.message(**msg_kwargs)
   
               # For `GNNExplainer`, we require a separate message and aggregate
               # procedure since this allows us to inject the `edge_mask` into the
               # message passing computation scheme.
               if self.__explain__:
                   edge_mask = self.__edge_mask__.sigmoid()
                   # Some ops add self-loops to `edge_index`. We need to do the
                   # same for `edge_mask` (but do not train those).
                   if out.size(self.node_dim) != edge_mask.size(0):
                       loop = edge_mask.new_ones(size[0])
                       edge_mask = torch.cat([edge_mask, loop], dim=0)
                   assert out.size(self.node_dim) == edge_mask.size(0)
                   out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
   
               aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
               out = self.aggregate(out, **aggr_kwargs)
   
               update_kwargs = self.inspector.distribute('update', coll_dict)
               return self.update(out, **update_kwargs)
           
           
       def forward(self, x, edge_index):
           # x has shape [N, in_channels]
           # edge_index has shape [2, E]
   
           # Step 1: Add self-loops to the adjacency matrix.
           edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
   
           # Step 2: Linearly transform node feature matrix.
           x = self.lin(x)
   
           # Step 3: Compute normalization.
           row, col = edge_index
           deg = degree(col, x.size(0), dtype=x.dtype)
           deg_inv_sqrt = deg.pow(-0.5)
           # note: norm is in shape of (number of edge, )
           norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
           print("Get degree Shape: ", edge_index.shape)
           print("Norm Shape: ",norm.shape)
           
           # Step 4-5: Start propagating messages.
           # Convert edge index to a sparse adjacency matrix representation, with row = from nodes, col = to nodes. 
           # When value =  1 in adjacency matrix, it indicates two nodes are adjacent.
           # adjmat = SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.shape[1]))
           
           # 这里 adjacency matrix 的值从1 变成 normalization 的值，方便乘法计算
           adjmat = SparseTensor(row=edge_index[0], col=edge_index[1], value=norm)
           
           # 此处传的不再是edge_idex，而是SparseTensor类型的Adjancency Matrix
           return self.propagate(adjmat, x=x, norm=norm, deg=deg.view((-1, 1)))
   
   
       def message(self, x_j, norm, deg_i=1):
           # x_j has shape [E, out_channels]
           # deg_i has shape [E, 1]
           # Step 4: Normalize node features.
           return norm.view(-1, 1) * x_j * deg_i
   
       def aggregate(self, inputs, index, ptr, dim_size):
           print('self.aggr:', self.aggr)
           print("`aggregate` is called")
           return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
   
       def message_and_aggregate(self, adj_t, x, norm,deg):
           # note: 
           # adj_t: adjacency matrix
           # norm: normalization coefficient 1/sqrt(deg_i)*sqrt(deg_j)
           # number of '1' in adj_t = length of norm
           
           ## Print something to debug
           #print('`message_and_aggregate` is called')
           #print("adj_t: ",adj_t)
           #print("deg:", deg)
           
           adj_t = adj_t.to_dense()
           N = len(adj_t)
           out = []
           x0 = x[:]
           for i in range(N):
               # 计算每个 xi 的neighbor传过来的信息的平均值
               x_sum = torch.matmul(x.T,adj_t[i])
               x_avg = x_sum/deg[i]
               out.append(x_avg)
           out = torch.stack(out)
           return [out, x0]
   
       def update(self, inputs, deg):
           print("Update result")
           print("Degree",deg)
           # resnet的结构
           x0 = inputs[1]
           output = self.relu(inputs[0]) + x0
           return output
   
   
   dataset = Planetoid(root='dataset/Cora', name='Cora')
   data = dataset[0]
   
   net = GCNConv(data.num_features, 64)
   h_nodes = net(data.x, data.edge_index)
   ```

## 参考资料

1. [CREATING MESSAGE PASSING NETWORKS](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#creating-message-passing-networks)
2. [torch_geometric.nn.conv.message_passing.MessagePassing](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.message_passing.MessagePassing)
3. [The “MessagePassing” Base Class](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#the-messagepassing-base-class)
4. [Implementing the GCN Layer](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#implementing-the-gcn-layer)



