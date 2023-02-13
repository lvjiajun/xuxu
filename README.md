# Pan12 抄袭检测匹配项目

> bert_keras 以bert4keras为框架的训练生成框架

> Generate_data 生成匹配数据

> huggingface 以huggingface 框架的训练生成框架

## 安装环境
### 新建conda环境
打开conda 命令终端，输入指令，这里以工程目录xuxu为例子
    conda create --name xuxu python==3.8
### 切换conda环境
    conda activate xuxu
### 安装pip 依赖
    pip install requirement.txt

## 出现常见问题
### 出现需要修改tensorflow 版本错误
异常错误：AttributeError: module 'tensorflow.python.framework.ops' has no attribute '_TensorLike'提示tensorflow 版本不对
解决方法：设置环境变量，在文本的第一段引入os后加入 

    import os 
    os.environ['TF_KERAS'] = '1'