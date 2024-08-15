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

### 本地数据集
这样修改后，用户就可以选择使用本地数据集或 Hugging Face 的数据集。使用方法如下：

使用 Hugging Face 数据集（默认）：

python tools/train_net.py
使用本地数据集：

python tools/train_net.py --use_local_data --train_file path/to/train.csv --val_file path/to/val.csv
需要注意的几点：

本地 CSV 文件应该包含 'text' 列（对于单句任务）或 'sentence1' 和 'sentence2' 列（对于句对任务），以及 'label' 列。

如果本地数据集的结构与预设的不同，你可能需要进一步修改 preprocess_function 来适应你的数据结构。

这个实现假设本地数据集是 CSV 格式的。如果你的数据集是其他格式（如 JSON 或 TSV），你需要相应地修改加载数据的代码。

对于大型数据集，你可能需要考虑使用 Hugging Face 的 datasets.Dataset.from_pandas() 方法来逐步加载数据，以避免内存问题。

通过这些修改，你的模板现在可以灵活地处理both Hugging Face 数据集和本地数据集，使其更加通用和实用。