# Language-Specific Neurons Analysis

Identify and analyze language-specific neurons in multilingual language models by tracking neuron activation patterns across different languages.

## Overview

This repository extends the original [Language-Specific Neurons](https://github.com/RUCAIBox/Language-Specific-Neurons) work with:
- Support for the mOscar dataset
- Extended compatibility with Qwen2.5 model family
- Tools for analyzing neuron activation patterns

## Note

Use VLLM V0 engine to run our code by exporting the environment variable
```
export VLLM_USE_V1=0
```