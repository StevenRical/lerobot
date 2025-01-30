<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="media/lerobot-logo-thumbnail.png">
    <source media="(prefers-color-scheme: light)" srcset="media/lerobot-logo-thumbnail.png">
    <img alt="LeRobot, Hugging Face Robotics Library" src="media/lerobot-logo-thumbnail.png" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/nightly-tests.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/nightly-tests.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/huggingface/lerobot/branch/main/graph/badge.svg?token=TODO)](https://codecov.io/gh/huggingface/lerobot)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Examples](https://img.shields.io/badge/Examples-green.svg)](https://github.com/huggingface/lerobot/tree/main/examples)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://dcbadge.vercel.app/api/server/C5P34WJ68S?style=flat)](https://discord.gg/s3KuuzsPFb)

</div>

<h2 align="center">
    <p><a href="https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md">New robot in town: SO-100</a></p>
</h2>

<div align="center">
    <img src="media/so100/leader_follower.webp?raw=true" alt="SO-100 leader and follower arms" title="SO-100 leader and follower arms" width="50%">
<p>我们刚刚添加了一个关于如何构建更实惠的机器人的新教程，每只手臂的价格为 110 美元！</p>
    <p>通过仅使用笔记本电脑向其展示一些动作来教它新技能。</p>
    <p>然后观看您自制的机器人自主行动🤯</p>
    <p>点击链接访问<a href="https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md">SO-100 完整教程</a>。</p >
</div>

<br/>

<h3 align="center">
   <p>LeRobot：用于现实世界机器人的最先进的人工智能</p>
</h3>

---

🤗 LeRobot 旨在为 PyTorch 中的现实世界机器人提供模型、数据集和工具。目标是降低进入机器人技术的门槛，以便每个人都可以做出贡献并从共享数据集和预训练模型中受益。

🤗 LeRobot 包含最先进的方法，这些方法已被证明可以转移到现实世界，重点是模仿学习和强化学习。
🤗 乐机器人已经提供了一组预训练模型、包含人类收集演示的数据集以及无需组装机器人即可开始使用的模拟环境。在未来几周内，该计划将在最实惠、功能最强大的机器人上增加对现实世界机器人技术的越来越多的支持。

🤗 LeRobot 在这个 Hugging Face 社区页面上托管预训练的模型和数据集：[huggingface.co/lerobot](https://huggingface.co/lerobot)

#### 模拟环境中预训练模型的示例

  <tr>
    <td><img src="media/gym/aloha_act.gif" width="100%" alt="ACT policy on ALOHA env"/></td>
    <td><img src="media/gym/simxarm_tdmpc.gif" width="100%" alt="TDMPC policy on SimXArm env"/></td>
    <td><img src="media/gym/pusht_diffusion.gif" width="100%" alt="Diffusion policy on PushT env"/></td>
  </tr>
  <tr>
<tdalign="center">关于 ALOHA env 的 ACT 政策</td>
    <tdalign="center">SimXArm 环境上的 TDMPC 策略</td>
    <tdalign="center">PushT 环境的扩散策略</td>
  </tr>
</table>

### 致谢

-感谢 Tony Zhao、Zipeng Fu 和同事开源 ACT 政策、ALOHA 环境和数据集。我们的改编自[ALOHA](https://tonyzhaozh.github.io/aloha)和[Mobile ALOHA](https://mobile-aloha.github.io)。

-感谢 Cheng Chi、Zhenjia Xu 及其同事开源 Diffusion 策略、Pusht 环境和数据集以及 UMI 数据集。我们的政策改编自[Diffusion Policy](https://diffusion-policy.cs.columbia.edu)和[UMI Gripper](https://umi-gripper.github.io)。

-感谢 Nicklas Hansen、Yunhai Feng 及其同事开源 TDMPC 策略、Simxarm 环境和数据集。我们的改编自[TDMPC](https://github.com/nicklashansen/tdmpc)和[FOWM](https://www.yunhaifeng.com/FOWM)。

-感谢 Antonio Loquercio 和 Ashish Kumar 的早期支持。

-感谢 [Seungjae (Jay) Lee](https://sjlee.cc/)、[Mahi Shafiullah](https://mahis.life/) 和同事开源 [VQ-BeT](https://sjlee.cc/vq-bet/) 政策并帮助我们使代码库适应我们的存储库。该政策改编自[VQ-BeT repo](https://github.com/jayLEE0301/vq_bet_official)。


＃＃ 安装

下载我们的源代码：
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

使用Python 3.10创建一个虚拟环境并激活它，例如与 [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html)：
```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

安装🤗 lerobot：
```bash
pip install -e .
```

> **注意：**根据您的平台，如果您在此步骤中遇到任何构建错误
您可能需要安装“cmake”和“build-essential”来构建我们的一些依赖项。
在 Linux 上：`sudo apt-get install cmake build-essential`

对于模拟，🤗lerobot配备了可以作为额外安装的环境：
- [aloha](https://github.com/huggingface/gym-aloha)
- [xarm](https://github.com/huggingface/gym-xarm)
- [pusht](https://github.com/huggingface/gym-pusht)

例如，要安装带有 aloha 和 Pusht 的 🤗 LeRobot，请使用：
```bash
pip install -e ".[aloha, pusht]"
```

要使用[权重和偏差](https://docs.wandb.ai/quickstart)进行实验跟踪，请使用以下命令登录
```bash
wandb login
```

（注意：您还需要在配置中启用 WandB。请参见下文。）

## 演练

```
.
├── examples             # 包含演示示例，从这里开始学习 LeRobot
|   └── advanced         # 包含更多示例，适合已经掌握基础知识的用户
├── lerobot
|   ├── configs          # 包含 hydra 的 YAML 文件，所有选项都可以在命令行中覆盖
|   |   ├── default.yaml   # 默认选择，加载 pusht 环境和 diffusion 策略
|   |   ├── env            # 各种模拟环境及其数据集：aloha.yaml、pusht.yaml、xarm.yaml
|   |   └── policy         # 各种策略：act.yaml、diffusion.yaml、tdmpc.yaml
|   ├── common           # 包含类和工具
|   |   ├── datasets       # 各种人类演示数据集：aloha、pusht、xarm
|   |   ├── envs           # 各种模拟环境：aloha、pusht、xarm
|   |   ├── policies       # 各种策略：act、diffusion、tdmpc
|   |   ├── robot_devices  # 各种真实设备：dynamixel 电机、opencv 摄像头、koch 机器人
|   |   └── utils          # 各种工具
|   └── scripts          # 包含可通过命令行执行的功能
|       ├── eval.py                 # 加载策略并在环境中评估
|       ├── train.py                # 通过模仿学习和/或强化学习训练策略
|       ├── control_robot.py        # 远程操作真实机器人、记录数据、运行策略
|       ├── push_dataset_to_hub.py  # 将数据集转换为 LeRobot 格式并上传到 Hugging Face Hub
|       └── visualize_dataset.py    # 加载数据集并渲染其演示
├── outputs               # 包含脚本执行的结果：日志、视频、模型检查点
└── tests                 # 包含用于持续集成的 pytest 工具

```

### 可视化数据集

查看[示例 1](./examples/1_load_lerobot_dataset.py)，它演示了如何使用我们的数据集类自动从 Hugging Face 中心下载数据。

您还可以通过从命令行执行我们的脚本来本地可视化集线器上数据集的剧集：
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0

```

或者使用“root”选项和“--local-files-only”从本地文件夹中的数据集（在以下情况下，将在“./my_local_data_dir/lerobot/pusht”中搜索数据集）
```bash
# T方块
python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --root ./my_local_data_dir/pusht \
    --local-files-only 1 \
    --episode-index 0

# 装咖啡
python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/aloha_static_coffee \
    --root ./my_local_data_dir/aloha_state_coffee \
    --local-files-only 1 \
    --episode-index 3


```


它将打开“rerun.io”并显示相机流、机器人状态和动作，如下所示：

https://github-production-user-asset-6210df.s3.amazonaws.com/4681518/328035972-fd46b787-b532-47e2-bb6f-fd536a55a7ed.mov?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240505%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240505T172924Z&X-Amz-Expires=300&X-Amz-Signature=d680b26c532eeaf80740f08af3320d22ad0b8a4e4da1bcc4f33142c15b509eda&X-Amz-SignedHeaders=host&actor_id=24889239&key_id=0&repo_id=748713144


我们的脚本还可以可视化存储在远程服务器上的数据集。有关更多说明，请参阅“python lerobot/scripts/visualize_dataset.py --help”。

### `LeRobotDataset` 格式
“LeRobotDataset”格式的数据集使用起来非常简单。它可以从 Hugging Face 中心的存储库或本地文件夹中加载，例如： `dataset = LeRobotDataset("lerobot/aloha_static_coffee")` 并且可以像任何 Hugging Face 和 PyTorch 数据集一样被索引。例如，“dataset[0]”将从数据集中检索单个时间帧，其中包含观察结果和一个动作，作为准备输入模型的 PyTorch 张量。
“LeRobotDataset”的特殊性在于，我们可以通过将“delta_timestamps”设置为相对于索引帧的相对时间列表，根据其与索引帧的时间关系来检索多个帧，而不是通过其索引检索单个帧。框架。例如，使用 `delta_timestamps = {"observation.image": [-1, -0.5, -0.2, 0]}`，对于给定的索引，可以检索 4 帧：3 个“前一”帧 1 秒、0.5 秒、索引帧之前 0.2 秒以及索引帧本身（对应于 0 条目）。有关“delta_timestamps”的更多详细信息，请参阅示例 [1_load_lerobot_dataset.py](examples/1_load_lerobot_dataset.py)。

在幕后，“LeRobotDataset”格式使用了多种方法来序列化数据，如果您打算更密切地使用这种格式，这对于理解数据很有用。我们试图制作一种灵活而简单的数据集格式，涵盖强化学习和机器人技术、模拟和现实世界中存在的大多数类型的特征和特性，重点关注相机和机器人状态，但很容易扩展到其他类型的感官只要输入可以用张量表示即可。
以下是通过 `dataset = LeRobotDataset("lerobot/aloha_static_coffee")` 实例化的典型 `LeRobotDataset` 的重要细节和内部结构组织。确切的特征会随着数据集的不同而改变，但主要方面不会改变：

```
数据集属性：
├ hf_dataset: Hugging Face 数据集（基于 Arrow/Parquet）。典型特征示例：
│ ├ observation.images.cam_high（视频帧）：
│ │ 视频帧 = {'path': 指向 mp4 视频的路径, 'timestamp'（float32）：视频中的时间戳}
│ ├ observation.state（float32 列表）：例如，机械臂关节的位置
│ ... （更多观测值）
│ ├ action（float32 列表）：例如，机械臂关节的目标位置
│ ├ episode_index（int64）：此样本对应的实验集序号
│ ├ frame_index（int64）：此样本在实验集中的帧序号；每个实验集从 0 开始
│ ├ timestamp（float32）：样本在实验集中的时间戳
│ ├ next.done（布尔值）：是否是实验集的结束；每个实验集的最后一帧为 True
│ └ index（int64）：整个数据集中的全局索引

├ episode_data_index: 包含每个实验集起始帧和结束帧索引的两个张量
│ ├ from（1D int64 张量）：每个实验集的第一帧索引——形状为（实验集数量，），从 0 开始
│ └ to（1D int64 张量）：每个实验集的最后一帧索引——形状为（实验集数量，）

├ stats: 一个统计信息字典（最大值、平均值、最小值、标准差），对应数据集的每个特征，例如：
│ ├ observation.images.cam_high: {'max': 与数据维度相同的张量（例如，图像为 (c, 1, 1)，状态为 (c,) 等）}
│ ...

├ info: 一个包含数据集元数据的字典
│ ├ codebase_version（字符串）：用于记录创建数据集时的代码库版本
│ ├ fps（浮点数）：数据集记录/同步的每秒帧数
│ ├ video（布尔值）：是否将帧以 mp4 视频格式编码以节省空间，或者以 png 文件存储
│ └ encoding（字典）：如果是视频，记录使用 ffmpeg 编码视频时的主要选项

├ videos_dir（路径）：存储/访问 mp4 视频或 png 图像的目录
└ camera_keys（字符串列表）：用于访问数据集中相机特征的键（例如，["observation.images.cam_high", ...]）
```

“LeRobotDataset”的每个部分使用几种广泛使用的文件格式进行序列化，即：
-使用 Hugging Face 数据集库序列化到 parquet 存储的 hf_dataset
-视频以mp4格式存储以节省空间
-元数据存储在纯 json/jsonl 文件中
数据集可以从 HuggingFace 中心无缝上传/下载。要处理本地数据集，您可以使用“local_files_only”参数，并使用“root”参数指定其位置（如果它不在默认的“~/.cache/huggingface/lerobot”位置）。

### 评估预训练策略

查看[示例 2](./examples/2_evaluate_pretrained_policy.py)，它演示了如何从 Hugging Face hub 下载预训练策略，并在其相应的环境中运行评估。
查看[示例 2](./examples/2_evaluate_pretrained_policy.py)，它演示了如何从 Hugging Face hub 下载预训练策略，并在其相应的环境中运行评估。

我们还提供了一个功能更强大的脚本，可以在同一部署过程中并行评估多个环境。以下是 [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht) 上托管的预训练模型的示例：
```
python lerobot/scripts/eval.py \
    -p lerobot/diffusion_pusht \
    eval.n_episodes=10 \
    eval.batch_size=10
```

注意：训练您自己的策略后，您可以使用以下方法重新评估检查点：

```bash
python lerobot/scripts/eval.py -p {OUTPUT_DIR}/checkpoints/last/pretrained_model
```

有关更多说明，请参阅“python lerobot/scripts/eval.py --help”。

### 训练你自己的策略

查看[示例 3](./examples/3_train_policy.py)，它演示了如何使用我们的 Python 核心库来训练模型，[示例 4](./examples/4_train_policy_with_script.md) 演示了如何使用我们的训练来自命令行的脚本。

一般来说，您可以使用我们的训练脚本轻松训练任何策略。下面是一个在 Aloha 模拟环境中根据人类收集的轨迹训练 ACT 策略以执行插入任务的示例：

```bash
python lerobot/scripts/train.py \
    policy=act \
    env=aloha \
    env.task=AlohaInsertion-v0 \
    dataset_repo_id=lerobot/aloha_sim_insertion_human
```

实验目录是自动生成的，并将在您的终端中显示为黄色。它看起来像“outputs/train/2024-05-05/20-21-12_aloha_act_default”。您可以通过将此参数添加到 `train.py` python 命令来手动指定实验目录：
```bash
    hydra.run.dir=your/new/experiment/dir
```

在实验目录中会有一个名为“checkpoints”的文件夹，其结构如下：

```bash

检查点目录结构：
├── 000250 # 第 250 步训练的检查点目录
│ ├── pretrained_model # Hugging Face 预训练模型目录
│ │ ├── config.json # Hugging Face 预训练模型配置文件
│ │ ├── config.yaml # 整合后的 Hydra 配置文件
│ │ ├── model.safetensors # 模型权重文件
│ │ └── README.md # Hugging Face 模型说明文档
│ └── training_state.pth # 优化器/调度器/RNG 状态以及训练步骤信息
```

要从检查点恢复训练，您可以将这些添加到 `train.py` python 命令中：
```bash
    hydra.run.dir=your/original/experiment/dir resume=true
```

它将加载预训练模型、优化器和调度器状态进行训练。有关更多信息，请参阅我们关于训练恢复的教程[此处](https://github.com/huggingface/lerobot/blob/main/examples/5_resume_training.md)。

要使用 wandb 记录训练和评估曲线，请确保您已将“wandb 登录”作为一次性设置步骤运行。然后，在运行上面的训练命令时，通过添加以下内容在配置中启用 WandB：

```bash
    wandb.enable=true
```

运行的 wandb 日志的链接也会在终端中以黄色显示。以下是它们在浏览器中的外观示例。另请查看[此处](https://github.com/huggingface/lerobot/blob/main/examples/4_train_policy_with_script.md#tropical-logs-and-metrics)，了解日志中一些常用指标的说明。

![](media/wandb.png)

注意：为了提高效率，在训练期间，每个检查点都会在少量的剧集上进行评估。您可以使用“eval.n_episodes=500”来评估比默认值更多的剧集。或者，训练后，您可能想要重新评估更多剧集的最佳检查点或更改评估设置。有关更多说明，请参阅“python lerobot/scripts/eval.py --help”。

#### 再现最先进的 (SOTA)

我们组织了我们的配置文件（在 [`lerobot/configs`](./lerobot/configs) 下找到），以便它们在各自的原始作品中重现给定模型变体的 SOTA 结果。只需运行：

```bash
python lerobot/scripts/train.py policy=diffusion env=pusht
```

重现 PushT 任务上扩散策略的 SOTA 结果。

预训练策略以及复制详细信息可以在 https://huggingface.co/lerobot 的“模型”部分找到。

＃＃ 贡献

如果您想为 🤗 LeRobot 做出贡献，请查看我们的[贡献指南](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md)。

### 添加新数据集

要将数据集添加到中心，您需要使用写入访问令牌登录，该令牌可以从 [Hugging Face 设置](https://huggingface.co/settings/tokens) 生成：
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

然后指向您的原始数据集文件夹（例如“data/aloha_static_pingpong_test_raw”），然后使用以下命令将数据集推送到集线器：
```bash
python lerobot/scripts/push_dataset_to_hub.py \
--raw-dir data/aloha_static_pingpong_test_raw \
--out-dir data \
--repo-id lerobot/aloha_static_pingpong_test \
--raw-format aloha_hdf5
```

See `python lerobot/scripts/push_dataset_to_hub.py --help` for more instructions.

如果您的数据集格式不受支持，请通过复制以下示例在 `lerobot/common/datasets/push_dataset_to_hub/${raw_format}_format.py` 中实现您自己的数据集格式[pusht_zarr](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/pusht_zarr_format.py), [umi_zarr](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/umi_zarr_format.py), [aloha_hdf5](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/aloha_hdf5_format.py), or [xarm_pkl](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/xarm_pkl_format.py).


### 添加预训练策略

训练完策略后，您可以使用类似于“${hf_user}/${repo_name}”的集线器 ID 将其上传到 Hugging Face 中心（例如 [lerobot/diffusion_pusht](https://huggingface.co/lerobot /diffusion_pusht））。

您首先需要找到位于实验目录中的检查点文件夹（例如“outputs/train/2024-05-05/20-21-12_aloha_act_default/checkpoints/002500”）。其中有一个“pretrained_model”目录，其中应包含：
-`config.json`：策略配置的序列化版本（遵循策略的数据类配置）。
-`model.safetensors`：一组 `torch.nn.Module` 参数，以 [Hugging Face Safetensors](https://huggingface.co/docs/safetensors/index) 格式保存。
-`config.yaml`：包含策略、环境和数据集配置的整合 Hydra 训练配置。策略配置应与“config.json”完全匹配。环境配置对于任何想要评估您的策略的人都很有用。数据集配置只是作为可重复性的书面记录。

要将这些上传到中心，请运行以下命令：
```bash
huggingface-cli upload ${hf_user}/${repo_name} path/to/pretrained_model
```

有关其他人如何使用您的政策的示例，请参阅 [eval.py](https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/eval.py)。


### 通过分析改进您的代码

用于分析策略评估的代码片段示例：
```python
from torch.profiler import profile, record_function, ProfilerActivity

def trace_handler(prof):
    prof.export_chrome_trace(f"tmp/trace_schedule_{prof.step_num}.json")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=3,
    ),
    on_trace_ready=trace_handler
) as prof:
    with record_function("eval_policy"):
        for i in range(num_episodes):
            prof.step()
            # 将代码插入到配置文件中，可能是 eval_policy 函数的整个主体
```

## 引文

如果您愿意，您可以通过以下方式引用这项工作：
```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

此外，如果您正在使用任何特定的策略架构、预训练模型或数据集，建议引用该作品的原始作者，如下所示：

- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu)
```bibtex
@article{chi2024diffusionpolicy,
	author = {Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
	title ={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
	journal = {The International Journal of Robotics Research},
	year = {2024},
}
```
- [ACT or ALOHA](https://tonyzhaozh.github.io/aloha)
```bibtex
@article{zhao2023learning,
  title={Learning fine-grained bimanual manipulation with low-cost hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}
```

- [TDMPC](https://www.nicklashansen.com/td-mpc/)

```bibtex
@inproceedings{Hansen2022tdmpc,
	title={Temporal Difference Learning for Model Predictive Control},
	author={Nicklas Hansen and Xiaolong Wang and Hao Su},
	booktitle={ICML},
	year={2022}
}
```

- [VQ-BeT](https://sjlee.cc/vq-bet/)
```bibtex
@article{lee2024behavior,
  title={Behavior generation with latent actions},
  author={Lee, Seungjae and Wang, Yibin and Etukuru, Haritheja and Kim, H Jin and Shafiullah, Nur Muhammad Mahi and Pinto, Lerrel},
  journal={arXiv preprint arXiv:2403.03181},
  year={2024}
}
```
