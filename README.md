# MiniMax-H3

昇腾 NPU 910C 上的 **MiniMax-H3 Ref2VA** 推理优化，将生成10s 1080P视频，从耗时17分钟，加速到3分钟以内，大幅提示生成效率，还能保证生成效果基本不下降。

## 生成效果

[生成效果](https://www.bilibili.com/video/BV1LJbw6uEtc/) 类似如下：

![1080p_img2](Ref2VA/CANN-9.1.0/INI8/assets/1080p_img2.png)

## 可选方案

### 方案一：ComyUI量化版适配

将ComfyUI在英伟达GPU上运行的量化模型，适配到华为昇腾910C上，推理速度8分钟可以生成一个10s时长的1080P视频，**Ref2VA · CANN 9.1.0 · INT8 · v1.1**（16 die）：

```text
Ref2VA/CANN-9.1.0/INI8/
```

- 镜像：`nuvic/minimax-h3:ref2va-cann9.1.0-int8-v1.1`
- 详见 `[Ref2VA/CANN-9.1.0/INI8/README.md](Ref2VA/CANN-9.1.0/INI8/README.md)`

