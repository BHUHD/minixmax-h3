# MiniMax-H3

[![](assets/README/Author.svg)]() [![](assets/README/Organization.svg)]() [![](assets/README/License.svg)]() [![](assets/README/Chat.svg)](https://gitter.im/leaveking/xunsi-note) [![](assets/README/Time.svg)]() 

## 简介

本项目，是将MiniMax H3视频生成模型，针对GPU/NPU类型，进行加速调优，达到最佳效率，并部署到Docker容器内，**下载模型和Docker镜像即可运行起来，对外提供API服务**。

## 英伟达GPU

在英伟达GPU上，运行MiniMax H3视频生成模型：

| 设备     | 显存    | 模型        | 10s 1080P生成时间 | 详情                                                         |
| -------- | ------- | ----------- | ----------------- | ------------------------------------------------------------ |
| RTX 5080 | 1卡*16G | FL2VA NVFP4 | 17分钟            | [FL2VA-RTX5080-NVFP4-CUDA13.2/readme.md](./FL2VA-RTX5080-NVFP4-CUDA13.2/readme.md) |

## 昇腾NPU

在华为昇腾NPU上，运行MiniMax H3视频生成模型：

| 设备 | 显存     | 模型        | 10s 1080P生成时间 | 详情                                                         |
| ---- | -------- | ----------- | ----------------- | ------------------------------------------------------------ |
| 910C | 8卡*128G | Ref2VA INI8 | 8分钟             | [Ref2VA/CANN-9.1.0/INI8/README.md](Ref2VA/CANN-9.1.0/INI8/README.md) |

