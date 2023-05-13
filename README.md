# 深度残差网络(Deep residual network, ResNet)

### 开发环境(Development Environment 개발환경)
- PyCharm
- Python
- Numpy
- Tensorflow

### 主要原理(주요원리)
深度残杀网络（Deep residual network, ResNet）的提出是CNN图像史上的一件里程碑事件，ResNet在ILSVRC和COCO 2015上的战绩：
![v2-5e98ec97def099a4e6fb6b7ce3b1d460_r](https://user-images.githubusercontent.com/60682087/131846437-fa958b7b-cd50-4eff-8f74-c9e7269c6bb4.jpg)

图1 ResNet在ILSVTRC和COCO 2015上的战绩

ResNet取得了5项第一，并又一次刷新了CNN模型在ImageNet上的历史：

![v2-606573bdaaa97de6b8b10fb00f76d29a_r](https://user-images.githubusercontent.com/60682087/131846932-f7042051-82f9-41ff-a0a8-8eb329c2e4ce.png)

图2 ImageNet分类Top-5误差

ResNet是解决了深度CNN模型难训练的问题，从图2中可以看到14年的VGG才19层，而15年的ResNet多达152层，这在网络深度完全不是一个量级上，所以如果是第一眼看这个图的话，肯定会觉得ResNet是靠深度取胜。事实当然是这样，但是ResNet还有架构上的trick，这才使得网络的深度发挥出作用，这个trick就是残差学习（Residual learning）。下面详细讲述ResNet的理论及实现。

### 深度网络的退化问题

从经验来看，网络的深度对模型的性能至关重要，当增加网络层数后，网络可以进行更加复杂的特征模式的提取，所以当模型更深时理论上可以取得更好的结果，从图2中也可以看出网络越深而效果越好的一个实践证据。但是更深的网络其性能一定会更好。实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。这个现象可以在图3中直观看出来：56层的网络比20层网络效果还要差。这不会是过拟合问题，因为56层网络的训练误差同样高。我们知道深层网络存在着梯度消失或者爆炸的问题，这使得深度学习模型很难训练。但是现在已经存在一些技术手段BatchNorm来缓解这个问题。因此，出现深度网络的退化问题是非常令人诧异的。

![v2-dcf5688dad675cbe8fb8be243af5e1fd_r](https://user-images.githubusercontent.com/60682087/131848046-51482eb7-2508-4b14-9138-e80f3dbc7971.png)

图3 20层与56层网络在CIFAR-10上的误差

### 残差学习

深度网络的退化问题至少说明深度网络不容易训练。但是我们考虑这样一个事实：现在你有一个浅层网络，你想通过向上堆积新层来建立深层网络，一个极端情况是这些增加的层什么也不学习，仅仅复制浅层网络的特征，即这样新层是恒等映射（Identity mapping）。在这种情况下，深层网络应该至少和浅层网络性能一样，也不应该出现退化现象。好吧，你不得不承认肯定是目前的训练方法有问题，才使得深层网络很难去找到一个好的参数。

这个有趣的假设让何博士灵感爆发，他提出了残差学习来解决退化问题。对于一个堆积层结构（几层堆积而成）当输入为![4](https://user-images.githubusercontent.com/60682087/131849108-8715a1d0-6229-4e1f-b219-8a81b042d165.JPG)时其学习到的特征记为![5](https://user-images.githubusercontent.com/60682087/131849203-2acf1e5d-0981-4281-ac4e-1f9d4a6bdc3b.JPG), 现在我们希望其可以学习到残差![6](https://user-images.githubusercontent.com/60682087/131849338-fec86d20-6ea0-45ab-89ca-b34e0462c573.JPG), 这样其实原始的学习特征是![7](https://user-images.githubusercontent.com/60682087/131849458-c110f808-1a42-45f4-90dd-57eb90be7e2a.JPG)。之所以这样是因为残差学习相比原始特征直接学习更容易。当残差为0时，此时堆积层仅仅做了恒等映射，至少网络性能不会下降，实际上残差不会为0，这也会使得堆积层在输入特征基础上学习到新的特征，从而拥有更好的性能。残差学习的结构如图4所示。这有点类似与电路中的“短路”，所以是一种短路连接（shortcut connection）。

![8](https://user-images.githubusercontent.com/60682087/131849710-b124f502-a97e-45e6-9cb6-c59f1de23ee2.JPG)

图4  残差学习单元

为什么残差学习相对更容易，从直观上看残差学习需要学习的内容少，因为残差一般会比较小，学习难度小点。不过我们可以从数学的角度来分析这个问题，首先残差单元可以表示为：

![9](https://user-images.githubusercontent.com/60682087/131849839-5ba5768d-331d-444a-9e26-6ee50dcd7059.JPG)

其中![10](https://user-images.githubusercontent.com/60682087/131849892-1a7b6e6b-c229-435e-aa10-db059b84e085.JPG)和![11](https://user-images.githubusercontent.com/60682087/131849921-c5706fb5-6e22-4b9b-9b08-a72bbe226c50.JPG)分别表示的是第![12](https://user-images.githubusercontent.com/60682087/131850030-d2ecb676-58ff-42e9-8127-f698e73ec647.JPG)个残差单元的输入和输出，注意每个残差单元一般包含多层结构。![13](https://user-images.githubusercontent.com/60682087/131850095-6485ed5f-defd-4102-b474-ff84439f45e0.JPG)是残差函数，表示学习到的残差，而 ![14](https://user-images.githubusercontent.com/60682087/131850124-565521e2-202c-4126-acde-b09b040e92ed.JPG)表示恒等映射，![15](https://user-images.githubusercontent.com/60682087/131850170-b7761dfc-c67e-48a3-b732-4d6627d20f81.JPG)是ReLU激活函数。基于上式，我们求得从浅层![16](https://user-images.githubusercontent.com/60682087/131850205-6816f810-a96a-46f0-9d38-9794328af350.JPG)到深层![17](https://user-images.githubusercontent.com/60682087/131850261-e9fcebb6-f4e1-4c95-961e-eb768d22f597.JPG)的学习特征为：

![18](https://user-images.githubusercontent.com/60682087/131850338-ec70d3fc-1129-40dd-84e9-1f31c198fcc5.JPG)

利用链式规则，可以求得反向过程的梯度：

![19](https://user-images.githubusercontent.com/60682087/131850415-ac62f832-3149-4e09-b4ea-1741556d7405.JPG)

式子的第一个因子![20](https://user-images.githubusercontent.com/60682087/131850485-afeabe56-2fe8-4884-8206-3aeddf7f0663.JPG)表示的损失函数到达![21](https://user-images.githubusercontent.com/60682087/131850521-7fcdd5b9-2194-4497-a9dd-37e8b6fe6fd5.JPG)的梯度，小括号中的1表明短路机制可以无损地传播梯度，而另外一项残差梯度则需要经过带有weights的层，梯度不是直接传递过来的。残差梯度不会那么巧全为-1，而且就算其比较小，有1的存在也不会导致梯度消失。所以残差学习会更容易。要注意上面的推导并不是严格的证明。

### ResNet的网络结构

ResNet网络是参考了VGG19网络，在其基础上进行了修改，并通过短路机制加入了残差单元，如图5所示。变化主要体现在ResNet直接使用stride=2的卷积做下采样，并且用global average pool层替换了全连接层。ResNet的一个重要设计原则是：当feature map大小降低一半时，feature map的数量增加一倍，这保持了网络层的复杂度。从图5中可以看到，ResNet相比普通网络每两层间增加了短路机制，这就形成了残差学习，其中虚线表示feature map数量发生了改变。图5展示的34-layer的ResNet，还可以构建更深的网络如表1所示。从表中可以看到，对于18-layer和34-layer的ResNet，其进行的两层间的残差学习，当网络更深时，其进行的是三层间的残差学习，三层卷积核分别是1x1，3x3和1x1，一个值得注意的是隐含层的feature map数量是比较小的，并且是输出feature map数量的1/4。

![v2-7cb9c03871ab1faa7ca23199ac403bd9_720w](https://user-images.githubusercontent.com/60682087/131850649-a02b82fb-8353-4b70-9585-7bf99a358901.jpg)

图5 ResNet网络结构图

![v2-1dfd4022d4be28392ff44c49d6b4ed94_720w](https://user-images.githubusercontent.com/60682087/131850782-b215fbee-683f-40f3-9955-3d5eb7868f4e.jpg)

表1 不同深度的ResNet

面我们再分析一下残差单元，ResNet使用两种残差单元，如图6所示。左图对应的是浅层网络，而右图对应的是深层网络。对于短路连接，当输入和输出维度一致时，可以直接将输入加到输出上。但是当维度不一致时（对应的是维度增加一倍），这就不能直接相加。有两种策略：（1）采用zero-padding增加维度，此时一般要先做一个downsamp，可以采用strde=2的pooling，这样不会增加参数；（2）采用新的映射（projection shortcut），一般采用1x1的卷积，这样会增加参数，也会增加计算量。短路连接除了直接使用恒等映射，当然都可以采用projection shortcut。

![v2-0892e5423616c30f69ded61111b111c0_720w](https://user-images.githubusercontent.com/60682087/131850894-af196b26-382c-4693-87fa-aaf7f6d566ec.png)

图6 不同的残差单元

作者对比18-layer和34-layer的网络效果，如图7所示。可以看到普通的网络出现退化现象，但是ResNet很好的解决了退化问题。

![v2-ac88d9e118e3a85922188daba84f7efd_720w](https://user-images.githubusercontent.com/60682087/131850991-c6bd947a-9f1a-4559-ad4e-4ab3e3c39a6b.jpg)

图7 18-layer和34-layer的网络效果

最后展示一下ResNet网络与其他网络在ImageNet上的对比结果，如表2所示。可以看到ResNet-152其误差降到了4.49%，当采用集成模型后，误差可以降到3.57%。

![v2-0a2c8a209a221817f91c1f1728327beb_720w](https://user-images.githubusercontent.com/60682087/131851136-266a74a6-2d68-41a9-89a2-ed9bba59a447.jpg)

表2 ResNet与其他网络的对比结果

说一点关于残差单元题外话，上面我们说到了短路连接的几种处理方式，其实作者在文献[2]中又对不同的残差单元做了细致的分析与实验，这里我们直接抛出最优的残差结构，如图8所示。改进前后一个明显的变化是采用pre-activation，BN和ReLU都提前了。而且作者推荐短路连接采用恒等变换，这样保证短路连接不会有阻碍。感兴趣的可以去读读这篇文章。

![v2-4e0bf37ecad2f306fe09d32a2d37d908_720w](https://user-images.githubusercontent.com/60682087/131851219-5ee577d9-61bd-4721-bb51-de2f2dccf5f2.jpg)

图8 改进后的残差单元及效果

### 项目设置(Project Setup)

<pre><code>pip install --upgrade pip</code></pre>
<pre><code>pip install numpy</code></pre>
<pre><code>pip install tensorflow</code></pre>

- 这个项目是我为了重新学习深度残差网络(Deep residual network, ResNet)而做的项目（이 프로젝트는 내가 Deep residual network를 다시 공부하기위해서 만든 프로젝트입니다.）
