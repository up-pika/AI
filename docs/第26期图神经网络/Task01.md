---
title: 第26期-Task01 简单图论、环境配置与PyG库
date: 2021-06-15 11:30:23
tags: 
- 图论
- PyG库
categories:
- 图神经网络GNN
- 第26期集成学习
index_img:http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210412202410444.png
---

# 一、理论先知

## 1.1 图结构数据

### 1.1.1 图的表示

**定义一（图）**：

- 一个图被记为$\mathcal{G}=\{\mathcal{V}, \mathcal{E}\}$，其中 $\mathcal{V}=\left\{v\_{1}, \ldots, v\_{N}\right\}$是数量为$N=|\mathcal{V}|$ 的结点的集合， $\mathcal{E}=\left\{e\_{1}, \ldots, e\_{M}\right\}$ 是数量为 $M$ 的边的集合。
- 图用节点表示实体（entities ），用边表示实体间的关系（relations）。
- 节点和边的信息可以是**类别型**的（categorical），类别型数据的取值只能是哪一类别。一般称类别型的信息为**标签（label）**。
- 节点和边的信息可以是**数值型**的（numeric），类别型数据的取值范围为实数。一般称类别型的信息为**属性（attribute）**。
- 大部分情况中，节点含有信息，边也可能含有信息。

**定义二（图的邻接矩阵）**：

- 给定一个图$\mathcal{G}=\{\mathcal{V}, \mathcal{E}\}$，其对应的**邻接矩阵**被记为$\mathbf{A} \in\{0,1\}^{N \times N}$。$\mathbf{A}\_{i, j}=1$表示存在从结点$v\_i$到$v\_j$的边，反之表示不存在从结点$v\_i$到$v\_j$的边。

- 在**无向图**中，从结点$v\_i$到$v\_j$的边存在，意味着从结点$v\_j$到$v\_i$的边也存在。因而**无向图的邻接矩阵是对称的**。

- 在**无权图**中，**各条边的权重被认为是等价的**，即认为**各条边的权重为$1$**。

- 对于**有权图**，其对应的邻接矩阵通常被记为$\mathbf{W} \in\{0,1\}^{N \times N}$，其中$\mathbf{W}\_{i, j}=w\_{ij}$表示从结点$v\_i$到$v\_j$的边的权重。若边不存在时，边的权重为$0$。

  一个无向无权图的例子：

  ![无向无权图](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210614122142014.png)

  其邻接矩阵为：
  $$
  \mathbf{A}=\left(\begin{array}{lllll}
    0 & 1 & 0 & 1 & 1 \\
    1 & 0 & 1 & 0 & 0 \\
    0 & 1 & 0 & 0 & 1 \\
    1 & 0 & 0 & 0 & 1 \\
    1 & 0 & 1 & 1 & 0
    \end{array}\right)
  $$

### 1.1.2 图的属性

**定义三（结点的度，degree）**：

- 对于有向有权图，结点$v\_i$的出度（out degree）等于从$v\_i$出发的边的权重之和，结点$v\_i$的入度（in degree）等于从连向$v\_i$的边的权重之和。
- 无向图是有向图的特殊情况，结点的出度与入度相等。
- 无权图是有权图的特殊情况，各边的权重为$1$，那么结点$v\_i$的出度（out degree）等于从$v\_i$出发的边的数量，结点$v\_i$的入度（in degree）等于从连向$v\_i$的边的数量。
- 结点$v\_i$的度记为$d(v\_i)$，入度记为$d\_{in}(v\_i)$，出度记为$d\_{out}(v\_i)$。

**定义四（邻接结点，neighbors）**：

- **结点$v\_i$的邻接结点为与结点$v\_i$直接相连的结点**，其被记为**$\mathcal{N(v\_i)}$**。
- **结点$v\_i$的$k$跳远的邻接节点（neighbors with $k$-hop）**指的是到结点$v\_i$要走$k$步的节点（一个节点的$2$跳远的邻接节点包含了自身）。

**定义五（行走，walk）**：

- $walk(v\_1, v\_2) = (v\_1, e\_6,e\_5,e\_4,e\_1,v\_2)$，这是一次“行走”，它是一次从节点$v_1$出发，依次经过边$e\_6,e\_5,e\_4,e\_1$，最终到达节点$v\_2$的“行走”。
- 下图所示为$walk(v\_1, v\_2) = (v\_1, e\_6,e\_5,e\_4,e\_1,v\_2)$，其中红色数字标识了边的访问序号。
- 在“行走”中，节点是运行重复的。

<img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210508134652644.png" alt="walk" style="zoom:50%;" />

**定理六**：

- 有一图，其邻接矩阵为 $\mathbf{A}$, $\mathbf{A}^{n}$为邻接矩阵的$n$次方，那么$\mathbf{A}^{n}[i,j]$等于从结点$v\_i$到结点$v\_j$的长度为$n$的行走的个数。

**定义七（路径，path）**：

- “路径”是结点不可重复的“行走”。

**定义八（子图，subgraph）**：

- 有一图$\mathcal{G}=\{\mathcal{V}, \mathcal{E}\}$，另有一图$\mathcal{G}^{\prime}=\{\mathcal{V}^{\prime}, \mathcal{E}^{\prime}\}$，其中$\mathcal{V}^{\prime} \in \mathcal{V}$，$\mathcal{E}^{\prime} \in \mathcal{E}$并且$\mathcal{V}^{\prime}$不包含$\mathcal{E}^{\prime}$中未出现过的结点，那么$\mathcal{G}^{\prime}$是$\mathcal{G}$的子图。

**定义九（连通分量，connected component）**：

- 给定图$\mathcal{G}^{\prime}=\{\mathcal{V}^{\prime}, \mathcal{E}^{\prime}\}$是图$\mathcal{G}=\{\mathcal{V}, \mathcal{E}\}$的子图。记属于图$\mathcal{G}$但不属于$\mathcal{G}^{\prime}$图的结点集合记为$\mathcal{V}/\mathcal{V}^{\prime}$ 。如果属于$\mathcal{V}^{\prime}$的任意结点对之间存在至少一条路径，但不存在一条边连接属于$\mathcal{V}^{\prime}$的结点与属于$\mathcal{V}/\mathcal{V}^{\prime}$的结点，那么图$\mathcal{G}^{\prime}$是图$\mathcal{G}$的连通分量。

  <img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210508145204864.png" alt="连通分量实例" title="连通分量实例" style="zoom: 67%;" />

  左右两边子图都是整图的连通分量。

**定义十（连通图，connected graph）**：

- 当一个图只包含一个连通分量，即其自身，那么该图是一个连通图。

**定义十一（最短路径，shortest path）**：

- $v\_{s}, v\_{t} \in \mathcal{V}$ 是图$\mathcal{G}=\{\mathcal{V}, \mathcal{E}\}$上的一对结点，结点对$v\_{s}, v\_{t} \in \mathcal{V}$之间所有路径的集合记为$\mathcal{P}\_{\mathrm{st}}$。结点对$v\_{s}, v\_{t}$之间的最短路径$p\_{\mathrm{s} t}^{\mathrm{sp}}$为$\mathcal{P}\_{\mathrm{st}}$中长度最短的一条路径，其形式化定义为
  $$
  p_{\mathrm{s} t}^{\mathrm{sp}}=\arg \min _{p \in \mathcal{P}_{\mathrm{st}}}|p|
  $$
  其中，$p$表示$\mathcal{P}\_{\mathrm{st}}$中的一条路径，$|p|$是路径$p$的长度。

**定义十二（直径，diameter）**：

- 给定一个连通图$\mathcal{G}=\{\mathcal{V}, \mathcal{E}\}$，其直径为其所有结点对之间的最短路径的最小值，形式化定义为

$$
\operatorname{diameter}(\mathcal{G})=\max _{v_{s}, v_{t} \in \mathcal{V}} \min _{p \in \mathcal{P}_{s t}}|p|
$$

**定义十三（拉普拉斯矩阵，Laplacian Matrix）**：

- 给定一个图$\mathcal{G}=\{\mathcal{V}, \mathcal{E}\}$，其邻接矩阵为$A$，其拉普拉斯矩阵定义为$\mathbf{L=D-A}$，其中$\mathbf{D=diag(d(v_1), \cdots, d(v_N))}$。

**定义十四（对称归一化的拉普拉斯矩阵，Symmetric normalized Laplacian）**：

- 给定一个图$\mathcal{G}=\{\mathcal{V}, \mathcal{E}\}$，其邻接矩阵为$A$，其规范化的拉普拉斯矩阵定义为

$$
\mathbf{L=D^{-\frac{1}{2}}(D-A)D^{-\frac{1}{2}}=I-D^{-\frac{1}{2}}AD^{-\frac{1}{2}}}
$$

### 1.1.3 图的种类

- **同质图**（Homogeneous Graph）：只有一种类型的节点和一种类型的边的图。
- **异质图**（Heterogeneous Graph）：存在多种类型的节点和多种类型的边的图。
  <img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210516164150162.png" alt="异质图" style="zoom:30%;" />
- **二部图**（Bipartite Graphs）：节点分为两类，只有不同类的节点之间存在边。
  <img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210516164400658.png" alt="二部图" style="zoom:25%;" />

### 1.1.4 图结构数据上的机器学习

<img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210508171206912.png" alt="" style="zoom: 33%;" />

1. **节点预测**：预测节点的类别或某类属性的取值
   1. 例子：对是否是潜在客户分类、对游戏玩家的消费能力做预测
2. **边预测**：预测两个节点间是否存在链接
   1. 例子：Knowledge graph completion、好友推荐、商品推荐
3. **图的预测**：对不同的图进行分类或预测图的属性
   1. 例子：分子属性预测
4. **节点聚类**：检测节点是否形成一个社区
   1. 例子：社交圈检测
5. **其他任务**
   1. **图生成**：例如药物发现
   2. **图演变**：例如物理模拟
   3. ……

### 1.1.5 应用神经网络于图面临的挑战

在学习了简单的图论知识，我们再来回顾应用神经网络于图面临的挑战。

过去的深度学习应用中，我们主要接触的数据形式主要是这四种：**矩阵、张量、序列（sequence）和时间序列（time series）**，**它们都是规则的结构化的数据。然而图数据是非规则的非结构化的**，它具有以下的特点：

1. **任意的大小和复杂的拓扑结构；**
2. **没有固定的结点排序或参考点；**
3. **通常是动态的，并具有多模态的特征；**
4. **图的信息并非只蕴含在节点信息和边的信息中，图的信息还包括了图的拓扑结构。**

![](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210508111141393-1622014310446.png)

以往的深度学习技术是为规则且结构化的数据设计的，无法直接用于图数据。应用于图数据的神经网络，要求

- **适用于不同度的节点**；

- **节点表征的计算与邻接节点的排序无关**；

- **不但能够根据节点信息、邻接节点的信息和边的信息计算节点表征，还能根据图拓扑结构计算节点表征**。

  下面的图片展示了一个需要根据图拓扑结构计算节点表征的例子。图片中展示了两个图，它们同样有两黄、两蓝、两绿，共6个节点，因此它们的节点信息相同；假设边两端节点的信息为边的信息，那么这两个图有一样的边，即它们的边信息相同。但这两个图是不一样的图，它们的拓扑结构不一样。

![](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210607160411448.png)

# 二、实战练习

1. **配置程序运行环境**---环境配置有点麻烦，许多费心和时间
2. 学习**PyG中图数据的表示及其使用**，即学习PyG中`Data`类
3. 学习**PyG中图数据集的表示及其使用**，即学习PyG中`Dataset`类

### 2.1 作业

通过继承`Data`类实现一个类，专门用于表示“机构-作者-论文”的网络。该网络包含“机构“、”作者“和”论文”三类节点，以及“作者-机构“和“作者-论文“两类边。对要实现的类的要求：

1）用不同的属性存储不同节点的属性；

2）用不同的属性存储不同的边（边没有属性）；

3）逐一实现获取不同节点数量的方法。

```python
from enum import Enum, unique

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt


@unique
class MyLabel(Enum):
    dept = 0
    paper = 1
    author = 2

    @staticmethod
    def get_name(val):
        return MyLabel(val).name

class MyDataset(Data):
    def __init__(self, input_data, **kwargs):
        self.data = input_data
        # 构建数据集需要的参数
        x, edge_index, y, author_dept_edge_index, author_paper_edge_index = self.__create_datasets()
        self.author_dept_edge_index = author_paper_edge_index
        self.author_paper_edge_index = author_paper_edge_index
        super().__init__(x=x, edge_index=edge_index, y=y, **kwargs)

    def __add_node(self, node_label_list, node, label):
        if node_label_list.count((node, label)) == 0:
            # 添加节点
            node_label_list.append((node, label))

        node_index = node_label_list.index((node, label))

        # 返回节点集，节点索引
        return node_label_list, node_index

    def __create_datasets(self):
        node_label_list = []
        edge_index = None
        author_dept_edge_index = None
        author_paper_edge_index = None

        for row in self.data.values.tolist():
            # 取出三个节点数据
            dept = row[0]
            paper = row[1]
            author = row[2]

            # 添加节点，得到节点索引
            node_label_list, dept_index = self.__add_node(node_label_list, dept, MyLabel.dept.value)
            node_label_list, paper_index = self.__add_node(node_label_list, paper, MyLabel.paper.value)
            node_label_list, author_index = self.__add_node(node_label_list, author, MyLabel.author.value)

            # 构建作者与机构的关系
            author_dept_index = np.array([[author_index, dept_index],
                                          [dept_index, author_index]])

            author_dept_edge_index = np.hstack((author_dept_edge_index, author_dept_index)) \
                if author_dept_edge_index is not None else author_dept_index
            # 构建作者与论文的关系
            author_paper_index = np.array([[author_index, paper_index],
                                           [paper_index, author_index]])

            author_paper_edge_index = np.hstack((author_paper_edge_index, author_paper_index)) \
                if author_paper_edge_index is not None else author_paper_index

            # 添加边的索引
            edge_index = np.hstack((edge_index, author_dept_index)) if edge_index is not None else author_dept_index
            edge_index = np.hstack((edge_index, author_paper_index))

        nodes = [[node] for node, label in node_label_list]
        labels = [[label] for node, label in node_label_list]

        x = torch.tensor(nodes)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        y = torch.tensor(labels)
        return x, edge_index, y, author_dept_edge_index, author_paper_edge_index

    @property
    def dept_nums(self):
        return self.data['dept'].nunique()

    @property
    def author_nums(self):
        return self.data['author'].nunique()

    @property
    def paper_nums(self):
        return self.data['paper'].nunique()


if __name__ == '__main__':
    print("有2个作者，分别写了2个论文，来自同一个机构")
    # 有2个作者，分别写了2个论文，来自同一个机构
    input_data = pd.DataFrame([[1, 1, 1], [2, 2, 2]], columns=['dept', 'paper', 'author'])

    data = MyDataset(input_data)
    print("Number of dept nodes:", data.dept_nums)
    print("Number of author nodes:", data.author_nums)
    print("Number of paper nodes:", data.paper_nums)
    # 节点数量
    print("Number of nodes:", data.num_nodes)
    # 边数量
    print("Number of edges:", data.num_edges)
    # 此图是否包含孤立的节点
    print("Contains isolated nodes:", data.contains_isolated_nodes())
    # 此图是否包含自环的边
    print("Contains self-loops:", data.contains_self_loops())
    # 此图是否是无向图
    print("Is undirected:", data.is_undirected())

    plt.figure(figsize=(6, 6))
    G = nx.Graph()
    index = 0
    x = data.x.tolist()
    y = data.y.tolist()
    for x_name, y_label in zip(x, y):
        G.add_node(index, label=MyLabel.get_name(y_label[0])+'-'+str(x_name[0]))
        index += 1

    edge_index = [(i, j) for i, j in zip(data.edge_index.tolist()[0], data.edge_index.tolist()[1])]
    G.add_edges_from(edge_index)
    pos = nx.spring_layout(G, iterations=20)
    nx.draw(G, pos, edge_color="grey", node_size=500)  # 画图，设置节点大小
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=10)
    plt.show()
```



# 参考资料

- [Chapter 2 - Foundations of Graphs, Deep Learning on Graphs](https://cse.msu.edu/~mayao4/dlg_book/chapters/chapter2.pdf)

