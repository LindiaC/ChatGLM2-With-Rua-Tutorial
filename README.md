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
### 收集数据集
我们需要大量由你直接产生的数据。收集途径因人而异，如果你想训练Chatbot，可以从微信/QQ聊天记录入手。如果你想训练上下文续写能力，可以从作文/演讲稿/博客入手。下面以收集微信聊天记录为例。
#### 使用工具
Github上有很多开源的微信聊天记录导出工具。下面以[留痕](https://github.com/LC044/WeChatMsg)为例。    
>如果有需要，可以在手机端通过`我-设置-通用-聊天记录迁移与备份  (v8.0.44)` 先把手机的聊天记录迁移到电脑上。   

在[这里](https://github.com/LC044/WeChatMsg/releases)下载最新的Release。请注意与微信版本的对应。  
右键点击刚刚下载的exe文件，以管理员身份运行，按照提示操作。此处具体操作步骤及问题请参阅[“留痕”使用教程](https://blog.lc044.love/post/5)。
如果我们成功导出所有的聊天数据，可以在当前目录的`data/聊天记录` 下找到一个`messages.csv` 文件。之后我们只需要用到这个。  
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
这里我们只需要TalkerId, Type, SubType, IsSender, StrContent 这几列。通过我的观察，你本人发送的信息满足`Type=1,SubType=0,IsSender=1` 。如果前后两行在同一个聊天里，它们的TalkerId也是相同的。  
接下来我们需要清洗数据并制作数据集。一个简单的实现可以在上面的`build_dataset.ipynb`找到。
