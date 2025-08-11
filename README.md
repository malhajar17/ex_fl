# FlexAI Experiments

This repository provides **a set of different experiments** designed for you to try out and explore **FlexAI**. These experiments range from running your very first training job, to fine-tuning language, diffusion, and text-to-speech models using techniques like QLoRA and LoRA, as well as integrating FlexAI with other platforms, such as experiment trackers.

## Getting Started

### Prerequisites

**FlexAI CLI**: Install the FlexAI CLI by following the steps shown in the [Installing the FlexAI CLI](https://docs.flex.ai/cli/installation/) guide.

## The Experiments

The following table lists out the experiments available in this repository. Each experiment is designed to walk you through a specific use case and contains its required code, as well as detailed instructions on how to run it on FlexAI:

| No. | Section                                                                                                                          | Description                                                                                                                |
| --- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 1   | [A Simple Training Job on FlexAI](/experiments/running-a-simple-training-job/README.md)                                          | Step-by-step guide to get your first Training Job on FlexAI running                                                        |
| 2   | [A Simple Distributed Data Parallel (DDP) Training Job on FlexAI](/experiments/running-a-simple-training-job-with-ddp/README.md) | Demonstrates that you only need to add 2 flags in order to start a DDP Training Job                                        |
| 3   | [Resuming a Training Job from a Checkpoint](/experiments/continuing-a-training-job-from-a-checkpoint/README.md)                  | Learn how to resume a Training Job from a previously saved checkpoint                                                      |
| 4   | [Streaming Large Datasets During a Training Job](/experiments/streaming-datasets/README.md)                                      | Train a model on a large dataset using streaming                                                                           |
| 5   | [Training Job & Experiment Tracking](/experiments/integrating-a-experiment-tracker/README.md)                                    | Using Weights and Biases with FlexAI for _experiment tracking_                                                             |
| 6   | [Fine-Tuning a Language Model with QLoRA](/experiments/qlora-ft-on-a-language-model/README.md)                                   | Fine-tune a causal language model efficiently using QLoRA                                                                  |
| 7   | [Fine-Tuning a Diffusion Model with LoRA](/experiments/lora-ft-on-a-diffusion-model/README.md)                                   | Fine-tune a diffusion model efficiently using LoRA                                                                         |
| 8   | [Fine-Tuning a Text-to-Speech Model](/experiments/ft-on-a-tts-model/README.md)                                                   | Fine-tune a text-to-speech (TTS) model                                                                                     |
| 9   | [Fine-Tuning a language Model using _Flash Attention_](/experiments/flash-attention-ft-on-a-language-model/README.md)            | Fine-tune a causal language model efficiently using the [flash-attn package](https://github.com/Dao-AILab/flash-attention) |
| 10  | [Fine-Tuning a language Model with LlamaFactory](/experiments/llama-factory/README.md)                                           | Demonstrates how to use [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) with FlexAI                               |

---

## Keep in mind

### Notes on HuggingFace Accelerate Integration

This repository includes experiments that utilize the **HuggingFace Accelerate library** for efficient training.

The **FlexAI CLI** simplifies running training scripts by automatically determining the appropriate execution method:

- **`python`**: Used for single-accelerator training.
- **`torchrun`**: Automatically used for multi-accelerator distributed training.

If you're accustomed to using the **`accelerate launch`** command, you can seamlessly run the same scripts on FlexAI without modification. Simply provide the script to FlexAI, and it will handle execution.

As highlighted in the [Accelerate documentation](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch#using-accelerate-launch), the `accelerate launch` command is optional. Instead, the Accelerate functionality integrates directly into your script, making it compatible with other launchers like `torchrun`.

> [!NOTE]
>
> Unlike `accelerate launch`, `torchrun` does not use the YAML configuration file generated by `accelerate config`.
>
> If your training setup relies on specific configurations from the YAML file, you may need to adjust your script to explicitly define these settings using the `Accelerator` class.
>
> By doing so, you ensure seamless execution across different environments while maintaining flexibility for various training setups.
# ex_fl
