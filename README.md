# Hugging Face 项目模板

这是一个使用 Hugging Face 生态系统的项目模板，包括 datasets、transformers 和 Trainer 模块。

## 安装

```zsh
conda deactivate
conda remove -n tf --all -y
conda clean -a -f -y
cd ~/.cache/pip && sudo rm -rf *
cd ~
#pip install --upgrade --force-reinstall --no-deps --no-cache-dir xxx
```

```zsh
conda update -n base conda -y
conda create -n tf python=3.11 -y
conda activate tf
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install transformers
pip install accelerate
pip install datasets
conda install conda-forge::jupyterlab -y
```

https://huggingface.co/docs/transformers/main/en/custom_models
https://huggingface.co/docs/transformers/main/en/trainer

## docs 自动生成

1. 安装 Sphinx 和相关扩展

```zsh
pip install sphinx sphinx-autodoc sphinx-rtd-theme
```

2. 初始化 Sphinx 项目

```zsh
# 在项目根目录下创建 docs 文件夹
mkdir docs
cd docs

# 运行 sphinx-quickstart 初始化
sphinx-quickstart
```

3. 配置 conf.py
在 docs/source/conf.py 中添加必要的配置:

```python
# 添加扩展
extensions = [
    'sphinx.ext.autodoc',  # 自动生成API文档
    'sphinx.ext.napoleon',  # 支持 Google/NumPy 风格的文档字符串
    'sphinx.ext.viewcode', # 添加源代码链接
    'sphinx.ext.coverage', # 文档覆盖率检查
]

# 设置主题
html_theme = 'sphinx_rtd_theme'

# 添加源代码路径
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # 根据实际项目结构调整路径
```
4. 创建文档结构
在 docs/source 目录下创建以下文件：

```rst
# index.rst
Welcome to Your Project's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   api

# modules.rst
API Documentation
===============

.. automodule:: your_package_name
   :members:
   :undoc-members:
   :show-inheritance:
```

5. 编写规范的文档字符串
在你的 Python 代码中使用规范的文档字符串 ：

6. 生成文档

```zsh
cd docs
# 生成 HTML 文档
make html

# 生成 PDF 文档（需要安装 LaTeX）
make latexpdf
```


## 使用方法

运行训练脚本：

### 使用 Accelerate 进行加速训练

本项目模板支持使用 Hugging Face 的 Accelerate 库进行加速训练。你可以通过以下方式启用 Accelerate 的功能：

1. 单 GPU 训练：
python tools/train_net.py


2. 多 GPU 数据并行训练：
accelerate launch tools/train_net.py


3. 使用混合精度训练：
accelerate launch --mixed_precision fp16 tools/train_net.py


4. 分布式训练（例如，使用 4 个 GPU）：
accelerate launch --num_processes 4 tools/train_net.py


你可以根据需要组合这些选项。更多关于 Accelerate 的用法，请参考 [Accelerate 文档](https://huggingface.co/docs/accelerate/index)。
通过这些修改，我们的项目模板现在支持使用 Hugging Face 的 Accelerate 库进行加速训练。主要的变化包括：