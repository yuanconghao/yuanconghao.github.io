---
date: 2022-06-08 00:51:06
layout: resource
---

<font color="#FF1E1E"><b>对学习过程中的数据集、框架、模型、论文、代码等进行整理，提供统一入口。</b></font>
<font color="#31C6D4"><b>Keep Learning, Keep Growing, Keep Succeeding</b></font>

## GPT

### AIGC

| <div style="width: 110px;">Datasets</div> | Source | Description |
| ---- | ------ | ----------- |
| OPENAI-ChatGPT | <a style="text-decoration:underline;" href="https://openai.com/blog/chatgpt">https://openai.com/blog/chatgpt</a> | 文本生成、对话 |
| LLama | <a style="text-decoration:underline;" href="https://github.com/facebookresearch/llama">https://github.com/facebookresearch/llama</a> | Facebook大大语言模型 |
| Alpaca | <a style="text-decoration:underline;" href="https://github.com/tatsu-lab/stanford_alpaca">https://github.com/tatsu-lab/stanford_alpaca</a> | LLama改进版 |
| Chinese-LLaMA-Alpaca | <a style="text-decoration:underline;" href="https://github.com/ymcui/Chinese-LLaMA-Alpaca">https://github.com/ymcui/Chinese-LLaMA-Alpaca</a> | 中文LLaMA模型和指令精调的Alpaca大模型 |

### IMAGE

| <div style="width: 110px;">Datasets</div> | Source | Description |
| ---- | ------ | ----------- |
| DALL·E2 | <a style="text-decoration:underline;" href="https://openai.com/product/dall-e-2/">https://openai.com/product/dall-e-2</a> | 文字生成图片，效果还可以。 |
| Midjourney | <a style="text-decoration:underline;" href="https://www.midjourney.com/showcase/recent/">https://www.midjourney.com/</a> | 生成的图片比较精致。 |
| Stable Diffusion | <a style="text-decoration:underline;" href="https://huggingface.co/blog/stable_diffusion">stable_diffusion blog</a> | 多模态图片生成 |

### AUDIO

| <div style="width: 110px;">Datasets</div> | Source | Description |
| ---- | ------ | ----------- |
| Whisper | <a style="text-decoration:underline;" href="https://openai.com/research/whisper/">https://openai.com/research/whisper</a> | 语音识别，语音转文字。 |

## 数据集

### 计算机视觉(CV)

| <div style="width: 110px;">Datasets</div> | Source | Description |
| ---- | ------ | ----------- |
| MNIST | <a style="text-decoration:underline;" href="http://yann.lecun.com/exdb/mnist/">http://yann.lecun.com/exdb/mnist/</a> | 手写数字识别；计算机视觉入门级数据集，包含各种手写数字图片。 |
| Fashion-MNIST | <a style="text-decoration:underline;" href="https://github.com/zalandoresearch/fashion-mnist">https://github.com/zalandoresearch/fashion-mnist</a> | 服饰识别；MNIST数据集过于简单，Fashion-MNIST可替代MNIST数据集，作为机器学习与深度学习算法基准。 |
| ImageNet | <a style="text-decoration:underline;" href="http://www.image-net.org/">http://www.image-net.org</a> | 图像识别；大规模数据集，几大经典CNN模型，AlexNet、VGG、GoogleNet、ResNet在ILSVRC大赛数据集。 |
| MS-COCO | <a style="text-decoration:underline;" href="https://cocodataset.org">https://cocodataset.org</a> | 目标检测、语义分割、图像标题生成；大规模的数据集。 |
| CIFAR-10 | <a style="text-decoration:underline;" href="http://www.cs.toronto.edu/~kriz/cifar.html">http://www.cs.toronto.edu/~kriz/cifar.html</a> | 图像分类；10个类别，每个类别6000张图片，5w个训练图片、1w个测试图片。 |
| SVHN | <a style="text-decoration:underline;" href="http://ufldl.stanford.edu/housenumbers/">http://ufldl.stanford.edu/housenumbers/</a> | 目标检测、文字检测；街景门牌号数据集，来源于谷歌街景图片。 |
| Open Images | <a style="text-decoration:underline;" href="https://storage.googleapis.com/openimages/web/factsfigures_v7.html">https://storage.googleapis.com/</a> | 语义分割、目标检测、图像分类；V4-V7 |
| LAION-5B | <a style="text-decoration:underline;" href="https://laion.ai/blog/laion-5b/">https://laion.ai/blog/laion-5b/</a> | the largest, freely accessible multi-modal dataset that currently exists.(目前最大的多模态开源数据集) |

### 自然语言处理(NLP)

| <div style="width: 110px;">Datasets</div> | Source | Description |
| ---- | ------ | ----------- |
| ACL-IMDB | <a style="text-decoration:underline;" href="http://ai.stanford.edu/~amaas/data/sentiment/">http://ai.stanford.edu/~amaas/data/sentiment/</a> | 电影评论数据集；大规模情感二分类数据集。 |
| WordNet | <a style="text-decoration:underline;" href="https://wordnet.princeton.edu/">https://wordnet.princeton.edu/</a> | 英语词库数据集 |

## 模型

### 经典卷积神经网络模型(CNN)

| <div style="width: 110px;">Model</div> | Paper | Description |
| ---- | ------ | ----------- |
| LeNet-5 | <a style="text-decoration:underline;" href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">Gradient-Based Learning Applied to Document Recognition</a> | <a style="text-decoration:underline;" href="http://yann.lecun.com/exdb/lenet/index.html">Yann LeCun（杨立昆）官网LetNet-5介绍；</a>1998年提出的CNN模型，主要用于手写字体识别，目前的CNN模型都没有逃出LetNet-5的卷积、池化、全连接架构。顶级大牛！！！ |
| AlexNet | <a style="text-decoration:underline;" href="http://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf">ImageNet Classification with Deep Convolutional Neural Networks</a> | <a style="text-decoration:underline;" href="http://www.cs.toronto.edu/~kriz/">Alex官网AlexNet介绍；</a>2012年提出的CNN模型， ImageNet LSVRC-2010竞赛冠军，具有划时代意义，再次之前主要用传统机器学习方法SVM，此后，深度学习发展迅速。 |
| VGG | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1409.1556.pdf">Very Deep Convolutional Networks for Large-scale Image Recognition</a> | <a style="text-decoration:underline;" href="https://www.robots.ox.ac.uk/~karen/">Karen Simonyan</a>，ImageNet LSVRC-2014竞赛亚军，VGG结构简单，应用性强，广受喜爱。<a style="text-decoration:underline;" href="https://www.robots.ox.ac.uk/~vgg/research/very_deep/">VGG-16、VGG-19效果较好。</a> |
| GoogleNet | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1409.4842.pdf">Going deeper with convolutions</a> | ImageNet LSVRC-2014竞赛冠军，22层网络，Top5错误率比VGG低约0.6个百分点。（结构有点复杂，不如VGG通用） |
| ResNet | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1512.03385.pdf">Deep Residual Learning for Image Recognition</a> | 大名鼎鼎的残差神经网络，ImageNet LSVRC-2015竞赛冠军，152层残差网络结构，将Top5错误率降到3.57，已经超过人眼水平，此后ImageNet大赛不再举办。 |

### 目标检测模型(ObjectDetection)

| <div style="width: 110px;">Model</div> | Paper | Description |
| ---- | ------ | ----------- |
| R-CNN | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1311.2524.pdf">Rich feature hierarchies for accurate object detection and semantic segmentation</a> | Two Stage开山之作，深度学习与传统机器学习结合，先选中2000个候选区域，AlexNet提取特征向量，SVM二分类，识别区域是否有目标；训练回归器，选中区域目标位置。测试集上能达到58.5%准确率，当时的王者，缺点是慢，且空间消耗大。 |
| Fast R-CNN | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1504.08083.pdf">Fast R-CNN</a> | 基于R-CNN和SPPNets，进行模型改进。不需要再生成2000个候选区域，只需要特征提取一次，使用selective search生成2000个区域候选框，再CNN卷积，Rol池化形成特定长度特征向量，送入全连接FC，Softmax，输出定位信息。速度较R-CNN有提升，但依旧慢。 |
| Faster R-CNN | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1506.01497.pdf">Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks</a> | 结构上将特征抽取，region proposal提取，bbox regression（包围边框回归），分类都整合到了一个网络中，综合性能有较大提高，检测速度提升较大。 |
| YOLO v1 | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1506.02640.pdf">You Only Look Once: Unified, Real-Time Object Detection</a> | One Stage开山之作，将检测任务当做回归问题处理。优点是速度快，但精度下降。 |
| YOLO v2 | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1612.08242.pdf">YOLO9000: Better, Faster, Stronger</a> | YOLO v1改进，使用了新的网络模型Darknet-19，加入了BN层，起到正则化效果；使用了高分辨率分类器；带Anchor Boxes的卷积；对边框进行K-Means聚类，可以直接定位预测。速度快，准确率较YOLO V1有提升，精度比SSD差。 | 
| YOLO v3 | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1804.02767.pdf">YOLOv3: An Incremental Improvement</a> | YOLO v2改进，使用新网络结构DarkNet-53，使用逻辑回归替代softmax作为分类器，融合FPN，实现多尺度检测。比较经典，在速度和准确率上都有提升，性能较好。也是作者的封笔之作，最后作者的自述比较有意思。In closing, do not @ me.(quit twitter)[破涕为笑] | 
| YOLO v4 | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/2004.10934.pdf">YOLOv4: Optimal Speed and Accuracy of Object Detection</a> | 将CV界大量研究成果进行了集成，提出了一套目标检测框架：输入、骨干、特征融合、输出；速度和精度上都有较大提升。 | 
| YOLO v5 | <a style="text-decoration:underline;" href="https://github.com/ultralytics/yolov5">YOLOv5 Source</a> | YOLO4发布两个月后，Glenn Jocher发布YOLO5，只有框架源码，无论文。架构上无创新，但号称模型大小比YOLO4小了近90%，但速度与YOLOv4不分伯仲。 | 
| YOLO v6 | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/2209.02976.pdf">YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications</a> | [美团视觉智能部][3]研发的目标检测框架，致力于工业应用。专注于检测精度和推理效率。[Github/YOLOV6][2] | 
| YOLO v7 | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/2207.02696.pdf">YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors</a> | 总结就是目前(2022)模型最小、速度最快、精度最高的目标检测模型。[Github/YOLOV7][4] | 
| SSD | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1512.02325.pdf">SSD: Single Shot MultiBox Detector</a> | 是一种One Stage的检测模型，相比于R-CNN系列模型上要简单许多。其精度与Faster R-CNN相匹敌，而速度达到59FPS，速度上超过Faster R-CNN |

### 生成对抗网络(GAN)

<!-- 参考：[生成对抗网络GAN论文TOP 10][1] -->

| <div style="width: 110px;">Model</div> | Paper | Description |
| ---- | ------ | ----------- |
| DCGANs | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1511.06434.pdf">Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks</a> | 卷积层与GAN相结合；并讨论了GAN特征的可视化、潜在空间插值、利用判别器特征来训练分类器、评估结果等问题。 |
| Improved Techniques for Training GANs | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1606.03498.pdf">Improved Techniques for Training GANs</a> | 分析了DCGAN，改进GAN训练的技术。 |
| Conditional GANs | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1411.1784.pdf">Conditional Generative Adversarial Nets</a> | 条件GAN是最先进的GAN之一，论文展示了如何整合数据的类标签，从而使 GAN 训练更加稳定。 |
| PG-GAN | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1710.10196.pdf">Progressive Growing of GANs for Improved Quality, Stability, and Variation</a> | 作者表示，这种方式不仅稳定了训练，GAN 生成的图像也是迄今为止质量最好的。 |
| BigGAN | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1809.11096.pdf">Large Scale GAN Training for High Fidelity Natural Image Synthesis</a> | BigGAN模型基于ImageNet生成图像质量最高的模型之一。 |
| StyleGAN | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1812.04948.pdf">A Style-Based Generator Architecture for Generative Adversarial Networks</a> | StyleGAN 模型非常先进，利用了潜在空间控制。 |
| CycleGAN | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1703.10593.pdf">Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks</a> | CycleGAN更具体地处理了没有成对训练样本的image-to-image转换的情况。CycleGAN有很多很酷的应用，比如超分辨率，风格转换，例如将马的图像变成斑马。 |
| Pix2Pix | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1611.07004.pdf">Image-to-Image Translation with Conditional Adversarial Networks</a> | Pix2Pix是另一种图像到图像转换的GAN模型。Pix2Pix有很多很酷的应用，比如将草图转换成逼真的照片。 |
| StackGAN | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1612.03242.pdf">StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks</a> | 与Conditional GAN 和Progressively Growing GANs最为相似。StackGAN是从自然语言文本生成图像。（牛） |
| GAN | <a style="text-decoration:underline;" href="https://arxiv.org/pdf/1406.2661.pdf">Generative Adversarial Networks</a> | 定义了GAN框架，并讨论了“非饱和”损失函数。论文在MNIST、TFD和CIFAR-10图像数据集上对GAN的有效性进行了实验验证。 |

## 工具

### 标注工具

| <div style="width: 110px;">Name</div> | Source | Description |
| ---- | ------ | ----------- |
| labelImg | <a style="text-decoration:underline;" href="https://github.com/heartexlabs/labelImg">labelImg</a> | 开源的图像标注工具，标签可用于分类和目标检测。 |
| labelme | <a style="text-decoration:underline;" href="https://github.com/wkentaro/labelme">labelme</a> | 图像语义分割标注工具 | 
| VIA | <a style="text-decoration:underline;" href="https://www.robots.ox.ac.uk/~vgg/software/via/">VGG Image Annotator (VIA)</a> | VGG图像注释器，可对图像、音频、视频标注。 | 
| EasyDL | <a style="text-decoration:underline;" href="https://ai.baidu.com/easydl/">EasyDL</a> | 百度推出AI开发平台，可采集、标注、清洗、训练。 | 

### 分词工具
| <div style="width: 110px;">Name</div> | Source | Description |
| ---- | ------ | ----------- |
| jieba | <a style="text-decoration:underline;" href="https://github.com/fxsjy/jieba">jieba</a> | 中文分词（精确模式、全模式、搜索引擎模式）、标注组件 |


## 顶会

| <div style="width: 110px;">Conference</div> | Description | Field |
| ---- | ------ | ----------- |
| ACL | Association of Computational Linguistics，每年开；计算语言学/自然语言处理方面最好的会议 | 人工智能/计算语言学 |
| IJCAI | International Joint Conference on Artificial Intelligence, 人工智能领域顶级国际会议，论文接受率18％左右 | 人工智能 |
| AAAI | American Association for Artificial Intelligence, 美国人工智能学会AAAI的年会，该领域的顶级会议 | 人工智能 |
| PRICAI | Pacific Rim International Conference on Artificial Intelligence, 亚太人工智能国际会议 | 人工智能 |
| ECCV | European Conference on Computer Vision, 领域顶级国际会议，录取率25%左右，2年一次，中国大陆每年论文数不超过20篇 | 模式识别/计算机视觉/多媒体计算 |
| ICML | International Conference on Machine Learning, 领域顶级国际会议，录取率25%左右，2年一次，目前完全国内论文很少 | 模式识别/计算机学习 |
| NIPS | Neural Information Processing Systems, 领域顶级国际会议，录取率20%左右，每年一次，目前完全国内论文极少（不超过5篇） | 神经计算/机器学习 |
| ACM MM | ACM Multimedia Conference, 领域顶级国际会议，全文的录取率极低，Poster较容易 | 多媒体技术/数据压缩 |
| IEEE ICCV |  International Conference on Computer Vision, 领域顶级国际会议，录取率20%左右，2年一次，中国大陆每年论文数不超过10篇 | 计算机视觉/模式识别/多媒体计算 |
| IEEE CVPR |  International Conference on Computer Vision and Pattern Recognition, 领域顶级国际会议，录取率25%左右，每年一次，中国大陆每年论文数不超过20篇 | 模式识别/计算机视觉/多媒体计算 |
| IEEE ICIP | International conference on Image Processing, 图像处理领域最具影响力国际会议，一年一次 | 图像处理 |
| IEEE ICME | International Conference on Multimedia and Expo, 多媒体领域重要国际会议，一年一次 | 多媒体技术 |

## 优秀网站

### 优秀学习网站

| <div style="width: 110px;">网站</div> | Description | Field |
| ---- | ------ | ----------- |
| 书栈网 | <https://www.bookstack.cn> | 计算机领域相关书籍、文档资料，很齐全 |


### 优秀工具网站

| <div style="width: 110px;">网站</div> | Description | Field |
| ---- | ------ | ----------- |
| 即时工具 | [https://www.67tool.com](https://www.67tool.com/category/5f56fb0164935e78271fd5a1) | 工具集合网站，200多个工具，图片处理、视频处理、文档处理等。 |
| LaTeX公式编辑器 | [https://www.latexlive.com](https://www.latexlive.com/) | LaTeX公式编辑器 |
| CNN Explainer | [cnn-explainer](https://poloclub.github.io/cnn-explainer/) | Learn Convolutional Neural Network (CNN) in your browser! |




[1]: https://mp.weixin.qq.com/s/gH6b5zgvWArOSfKBSIG1Ww
[2]: https://github.com/meituan/YOLOv6
[3]: https://tech.meituan.com/2022/06/23/yolov6-a-fast-and-accurate-target-detection-framework-is-opening-source.html
[4]: https://github.com/WongKinYiu/yolov7