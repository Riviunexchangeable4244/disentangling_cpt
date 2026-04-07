# Disentangling Continued Pre-training: Attention-Driven Routing and Semantic Hub Preservation in Language Adaptation
Khanh-Tung Tran, Vinh-Khanh Tran, Barry O'Sullivan, Hoang D. Nguyen

*Accepted to ACL 2026*

## Overview

> Continued Pretraining (CPT) effectively enables Large Language Models (LLMs) to acquire new target-language capabilities, yet the mechanisms underlying this second-language adaptation remain poorly understood. In this work, we investigate how CPT adapts model representations to accommodate new languages. Our extensive experiments reveal that second-language abilities emerge through a **selective adaptation mechanism**: task-solving capabilities are preserved in the **semantic hub**, while **interface layers** retarget to accommodate shifted token distributions. Through layer-swapping experiments, we demonstrate that semantic understanding can be surgically transferred between base and CPT models while maintaining cross-lingual functionality (e.g., swapping 50% of the parameters reduces performance by only 0.7%). Furthermore, we establish that attention components route language adaptation: larger parameter changes than FFN, correlate more strongly with language-specific neurons, and their surgical replacement substantially degrades performance, unlike FFN. Overall, our work provides a mechanistic understanding, guiding future work on efficient strategies for language adaptation.

---

## Experiments

### I. CPT enables second-language abilities through a selective adaptation mechanism

#### 1. Compute Sentence Retrieval Accuracy

Evaluate cross-lingual sentence retrieval on base and CPT models:

```bash
python sentence-retrieval/sr_experiment.py \
    --model_path <model_path> \
    --base <base_language> \
    --target <target_language>
```

#### 2. Determine Layer Specialization (Interface or Semantic Hub)

```bash
python layer-specialize/layer_specialize.py
```

#### 3. Benchmark Downstream Task and Correlate

Note: We extend [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to include `SIB200` task in `lm_eval_sib200` folder.

```bash
lm_eval --model hf \
    --model_args pretrained=<model> \
    --tasks <task>
```

```bash
python weight-diff/int_and_sem.py
```

Correlate results:

```bash
python downstream-correlation/downstream_correlation.py
```

#### 4. Swap Semantic Hub Layers and Re-evaluate

```bash
python layer-swap/layer-swap.py
```

Evaluate to verify:
- Swapping Semantic Hub layers exhibit low performance drop
- Swapping the same amount of layers randomly selected exhibit severe performance drop

```bash
python sentence-retrieval/sr_experiment.py \
    --model_path <swapped_model_path> \
    --base <base_language> \
    --target <target_language>
```



---

### II. Attention is the routing factor behind language adaptation during CPT

#### 1. Attention vs FFN Weight Differences

```bash
python weight-diff/attn_vs_ffn.py \
    --base-model <base-model> \
    --ckpt-model <ckpt-model>
```

#### 2. Language Specific Neuron Detection

##### Load mOSCAR Dataset

Tokenize multilingual OSCAR corpus for training:

```bash
python lsn/load_oscar.py \
    --languages en,zh,ga \
    --model-id model_name \
    --tokenizer path/to/tokenizer \
    --output-dir oscar_ids/
```

##### Identify Language-Specific Neurons

Get activation patterns:

```bash
python lsn/activation.py \
    -m <model> \
    -i <id_path> \
    -t <type> \
    -s <save_folder> \
    -n <name>
```

Detect LSNs by analyzing activation patterns:

```bash
python lsn/identify.py \
    -t <tag> \
    -l <lang>
```

#### 3. Language Specific Neuron Activation Correlation

##### Track LSN Activation Across CPT

Monitor LSN activations throughout the training trajectory:

```bash
python lsn/lsn_activation_cpt.py \
    --model <model> \
    --id_path <id_path> \
    --type <type> \
    --save_folder <save> \
    --name <name> \
    --mask_file <mask> \
    --lang <lang>
```

##### Analyze LSN Weight Changes

```bash
python weight-diff/gate_lsn.py \
    --base-model path/to/base \
    --target-model path/to/cpt \
    --mask-file masks.json \
    --lang target_lang \
    --output results/output.json
```

##### Correlate LSN Activation

Analyze correlation between:

* Attention weight changes in interface layers and LSN activation across checkpoints.
* LSN weight changes and LSN activation across checkpoints.

```bash
python3 activation-correlation/activation-correlation.py
```

#### 4. Component Swapping and Re-evaluation

##### Swap Attention in Interface Layers vs FFN in Layers with Many LSNs

```bash
python component-swap/attn_int_swap.py
python component-swap/ffn_lsn_swap.py
```

##### Evaluate Swapped Models

Run sentence retrieval on swapped models to verify:

* **Semantic preservation** for FFN swap
* **Degradation** for Attention swap
