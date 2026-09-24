# MiniMax-H3

[![](assets/README/Author.svg)]() [![](assets/README/Organization.svg)]() [![](assets/README/License.svg)]() [![](assets/README/Chat.svg)](https://gitter.im/leaveking/xunsi-note) [![](assets/README/Time.svg)]() 

## 简介

本项目，根据GPU型号对MiniMax H3视频生成模型的部署，进行有针对性的加速优化，使其达到最佳效率。项目提供了Docker镜像，**下载Docker镜像即可运行起来**，并对外提供API服务，即：

> 几乎零配置，即可在对应设备运行起来，效率还是最优。

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

