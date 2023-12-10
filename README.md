# ChatGLM2-With-Rua-Tutorial
**一个超级无敌保姆级教程：制作自己的数字克隆，0门槛0预算，可互动可分享。**  
---
**最低配置**：一台电脑，一个知道python是什么的脑子即可。**硬件方面不需要任何算力。**   
**灵感来自：[DK数字版](https://greatdk.com/1908.html#comment-5026)** 
一个最终成品的Demo可以在[这里](https://huggingface.co/spaces/Lindia/ChatWithRua)找到。  
![Demo](image\demo.png)  
*（你肯定会做得比我好，因为我的聊天数据集质量不高，我目前才经历了3年“手机不会被妈妈收掉可以自由聊天”的生活。）*
  
本教程分为两部分，第一部分为极速版过程描述，第二部分为笔者个人在制作过程中遇到的一些弯路与问题，以及解决方案。  
本文以[ChatGLM2-6b](https://github.com/THUDM/ChatGLM2-6B)为起点。
## 极速版过程
### 制作数据集
我们需要大量由你直接产生的数据。收集途径因人而异，如果你想训练Chatbot，可以从微信/QQ聊天记录入手。如果你想训练上下文续写能力，可以从作文/演讲稿/博客入手。下面以收集微信聊天记录为例。
#### 使用工具
Github上有很多开源的微信聊天记录导出工具。下面以[留痕](https://github.com/LC044/WeChatMsg)为例。    
>如果有需要，可以在手机端通过`我-设置-通用-聊天记录迁移与备份  (v8.0.44)` 先把手机的聊天记录迁移到电脑上。   

在[这里](https://github.com/LC044/WeChatMsg/releases)下载最新的Release。请注意与微信版本的对应。  
右键点击刚刚下载的exe文件，以管理员身份运行，按照提示操作。此处具体操作步骤及问题请参阅[“留痕”使用教程](https://blog.lc044.love/post/5)。
如果我们成功导出所有的聊天数据，可以在当前目录的`data/聊天记录` 下找到一个`messages.csv` 文件。之后我们只需要用到这个。  
#### 数据处理
`messages.csv`的形式如下：
```
localId,TalkerId,Type,SubType,IsSender,CreateTime,Status,StrContent,StrTime
12245,3,1,0,1,1528935478,,如果有帮助的话,2013-11-04 21:51:18,1234567890987654
12246,3,1,0,1,1572875478,,给我点个Star好不好😗,2019-11-05 22:47:11,123987654987654
```
先在第一行的末尾再加一个列名，使得每一列对齐，方便读入。
```
localId,TalkerId,Type,SubType,IsSender,CreateTime,Status,StrContent,StrTime,_
```
这里我们只需要`TalkerId, Type, SubType, IsSender, StrContent` 这几列。通过我的观察，你本人发送的信息满足`Type=1, SubType=0, IsSender=1` 。如果前后两行在同一个聊天里，它们的`TalkerId`也是相同的。  
接下来我们需要清洗数据并制作数据集。参考[ChatGLM2-6b](https://github.com/THUDM/ChatGLM2-6B)官方的训练示例，数据集格式应该长这个样子，`content`是问，`summary`是答：
```
{"content": "我会成为厉害的人吗", "summary": "肯定会的"}
{"content": "你喜欢Rua吗", "summary": "好喜欢♥♥"}
```
一个简单的实现可以在上面的`build_dataset.ipynb`找到。这个文件的功能包括：drop掉所有内容为空的消息；统计总消息数；按照`TalkerId`排序；找到所有你发送的长度大于5个字符的文本消息；找到它们的上一条；统计最终可用的消息数；生成需要的json文件。你可以根据需要自行修改。
之后我们就得到了两个文件：`train.json`和`dev.json`。这就是我们所需要的训练数据。  
### 开始训练
大模型对显存的要求比较高，`ChatGLM2-6b-int4`是一个降低显存的解决方案。使用Kaggle的两块Tesla T4时，每块占用8G显存。Kaggle免费方案可以充裕地满足我们的训练要求，因此这里以在Kaggle上运行为例。
首先我们下载[ChatGLM2-6b](https://github.com/THUDM/ChatGLM2-6B)。
```
git clone https://github.com/THUDM/ChatGLM-6B.git
```
clone好了之后进入`ChatGLM2-6B/ptuning` 文件夹，新建一个文件夹`WechatMsg`，把刚刚的`train.json`和`dev.json`放在下面。（当然你也可以随便指定路径，只要把这两个文件放进去就行。）  
然后我们进入Kaggle。前置知识：  
1. 如何创建一个Kaggle账号。
2. 如何上传自己的数据集。
3. 如何新建一个Kaggle笔记本、关联自己的数据集并开启GPU加速。

这些都可以非常方便地在网上学习到（比如[这里](https://blog.csdn.net/qq_53919099/article/details/130867160)）。
我们需要开启GPU T4 * 2。