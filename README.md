# ViDT: An Efficient and Effective Fully Transformer-based Object Detector

by [Hwanjun Song](https://scholar.google.com/citations?user=Ijzuc-8AAAAJ&hl=en&oi=ao)<sup>1</sup>, [Deqing Sun](https://scholar.google.com/citations?hl=en&user=t4rgICIAAAAJ)<sup>2</sup>, [Sanghyuk Chun](https://scholar.google.com/citations?user=4_uj0xcAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Varun Jampani](https://scholar.google.com/citations?hl=en&user=1Cv6Sf4AAAAJ)<sup>2</sup>, [Dongyoon Han](https://scholar.google.com/citations?user=jcP7m1QAAAAJ&hl=en)<sup>1</sup>, <br>[Byeongho Heo](https://scholar.google.com/citations?user=4_7rLDIAAAAJ&hl=en)<sup>1</sup>, [Wonjae Kim](https://scholar.google.com/citations?hl=en&user=UpZ41EwAAAAJ)<sup>1</sup>, and [Ming-Hsuan Yang](https://scholar.google.com/citations?hl=en&user=p9-ohHsAAAAJ)<sup>2,3</sup>

<sup>1</sup> NAVER AI Lab, <sup>2</sup> Google Research, <sup>3</sup> University California Merced

* **`Oct 8, 2021`:** **Our work is publicly available at [ArXiv](https://arxiv.org/abs/2110.03921).**
* **`Oct 18, 2021`:** **ViDT now supports [Co-scale conv-attentional image Transformers (CoaT)](https://arxiv.org/pdf/2104.06399.pdf) as another body structure.**
* **`Oct 22, 2021`:** **ViDT introduces and incorporates a cross-scale fusion module based on feature pyramid networks.**
* **`Oct 26, 2021`:** **[IoU-awareness loss](https://arxiv.org/pdf/1912.05992.pdf), and [token labeling loss](https://arxiv.org/pdf/2104.10858.pdf) are available with ViDT.**
* **`Nov 5, 2021`:** **The official code is released! We are uploading pre-trained models and it will takes a few days.**


## ViDT: Vision and Detection Transformers

### Highlight

<p align="center">
<img src="figures/binded_figure.png " width="900"> 
</p>

ViDT is an end-to-end fully transformer-based object detector, which directly produces predictions without using convolutional layers. Our main contributions are summarized as follows:

* ViDT introduces a modified attention mechanism, named **Reconfigured Attention Module (RAM)**, that facilitates any ViT variant to handling the appened [DET] and [PATCH] tokens for a standalone object detection. Thus, we can modify the lastest Swin Transformer backbone with RAM to be an object detector and obtain high scalability using its local attetention mechanism with linear complexity.

* ViDT adopts a **lightweight encoder-free neck** architecture to reduce the computational overhead while still enabling the additional optimization techniques on the neck module. As a result, ViDT obtains better performance than neck-free counterparts.

* We introdcue a new concept of **token matching for knowledge distillation**, which brings additional performance gains from a large model to a small model without compromising detection efficiency.

**Architectural Advantages**. First, ViDT enables to combine Swin Transformer and the sequent-to-sequence paradigm for detection. Second, ViDT can use the multi-scale features and additional techniques without a significant computation overhead. Therefore, as a fully transformer-based object detector, ViDT facilitates better integration of vision and detection transformers. 

**Component Summary**. There are four components: (1) RAM to extend Swin Transformer as a standalone object detector, (2) the neck decoder to exploit multi-scale features  with two additional techniques, auxiliary decoding loss and iterative box refinement, (3) knowledge distillation to benefit from a large model, and (4) decoding layer drop to further accelerate inference speed.

### Evaluation

**Index:** [[A. ViT Backbone](#A)], [[B. Main Results](#B)], [[C. Complete Analysis](#C)]
```
|--- A. ViT Backbone used for ViDT
|--- B. Main Results in the ViDT Paper
     |--- B.1. ViDT for 50 and 150 Epochs
     |--- B.2. Distillation with Token Matching
|--- C. Complete Component Analysis
```

<a name="A"></a>
#### A. [ViT Backbone used for ViDT](#content)

| Backbone and Size | Training Data | Epochs | Resulution | Params | ImageNet Acc. | Checkpoint |
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| `Swin-nano` | ImageNet-1K | 300 | 224 | 6M | 74.9% | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-swin/swin_nano_patch4_window7_224.pth) |
| `Swin-tiny` | ImageNet-1K | 300 | 224 | 28M | 81.2% | [Github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth) |
| `Swin-small` | ImageNet-1K | 300 | 224 | 50M | 83.2% | [Github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth) |
| `Swin-base` | ImageNet-22K | 90 | 224 | 88M | 86.3% | [Github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth) |

<a name="B"></a>
#### B. [Main Results in the ViDT Paper](#content)

In main experiments, auxiliary decoding loss and iterative box refinement were used as the auxiliary techniques on the neck structure. <br>
The efficiacy of distillation with token mathcing and decoding layer drop are verified independently in [Compelete Component Analysis](#C).<br>
`All the models were re-trained with the final version of source codes. Thus, the value may be very slightly different from those in the paper.`

##### B.1. VIDT for 50 and 150 epochs

| Backbone | Epochs | AP | AP50 | AP75 | AP_S | AP_M | AP_L | Params | FPS | Checkpoint / Log |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | 
| `Swin-nano` | 50 (150) | 40.4 (42.6) | 59.9 (62.2) | 43.0 (45.7) | 23.1 (24.9) | 42.8 (45.4) | 55.9 (59.1) | 16M | 20.0 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_nano_50.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_nano_50.txt) <br>([Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_nano_150.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_nano_150.txt))|
| `Swin-tiny` | 50 (150)| 44.9 (47.2) | 64.7 (66.7) | 48.3 (51.4) | 27.5 (28.4) | 47.9 (50.2) | 61.9 (64.7) | 38M | 17.2 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_tiny_50.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_tiny_50.txt) <br>([Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_tiny_150.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_tiny_150.txt))|
| `Swin-small` | 50 (150) | 47.4 (48.8) | 67.7 (68.8) | 51.2 (53.0) | 30.4 (30.7) | 50.7 (52.0) | 64.6 (65.9) | 60M | 12.1 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_small_50.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_small_50.txt) <br>([Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_small_150.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_small_150.txt))|
| `Swin-base` | 50 (150) | 49.4 (50.4) | 69.6 (70.4) | 53.4 (54.8) | 31.6 (34.1) | 52.4 (54.2) | 66.8 (67.4) | 0.1B | 9.0 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_base_50.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_base_50.txt) <br>([Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_base_150.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_base_150.txt)) |

##### B.2. Distillation with Token Matching (Coefficient 4.0)

All the models are trained for 50 epochs with distillation.

<table>
<thead>
<tr>
<th colspan="1">Teacher</th>
<th colspan="3">ViDT (Swin-base) trained for 50 epochs</th>
</tr>
<tr>
<th>Student</th>
<th>ViDT (Swin-nano)</th>
<th>ViDT (Swin-tiny)</th>
<th>ViDT (Swin-Small)</th>
</tr>
<td>Coefficient = 4.0</td>
<td>41.8</td>
<td>46.6</td>
<td>49.2</td>
</tr>
<tr>
<td>Checkpoint / Log</td>
<td> <a href="https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-distil-e50/vidt_nano_50.pth">Github</a> / <a href="https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-distil-e50/vidt_nano_50.txt">Log</a> </td>
<td> <a href="https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-distil-e50/vidt_tiny_50.pth">Github</a> / <a href="https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-distil-e50/vidt_tiny_50.txt">Log</a> </td>
<td> <a href="https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-distil-e50/vidt_small_50.pth">Github</a> / <a href="https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-distil-e50/vidt_small_50.txt">Log</a> </td>
</tr>
</thead>
<tbody>
</tr>
</tbody>
</table>


<a name="C"></a>
#### C. [Complete Component Analysis](#content)

We combined the four proposed components (even with distillation with token matching and decoding layer drop) to achieve high accuracy and speed for object detection. For distillation, ViDT (Swin-base) trained for 50 epochs was used for all models.

<table>
<thead>
<tr>
<th colspan="1"> </th>
<th colspan="4">Component</th>
<th colspan="3">Swin-nano</th>
<th colspan="3">Swin-tiny</th>
<th colspan="3">Swin-small</th>
</tr>
<tr>
<th>#</th>
<th>RAM</th>
<th>Neck</th>
<th>Distil</th>
<th>Drop</th>
<th>AP</th>
<th>Params</th>
<th>FPS</th>
<th>AP</th>
<th>Params</th>
<th>FPS</th>
<th>AP</th>
<th>Params</th>
<th>FPS</th>
</tr>
</thead>
<tbody>
<tr>
<td>(1)</td>
<td>:heavy_check_mark:</td>
<td></td>
<td></td>
<td></td>
<td>28.7</td>
<td>7M</td>
<td>36.5</td>
<td>36.3</td>
<td>29M</td>
<td>28.6</td>
<td>41.6</td>
<td>52M</td>
<td>16.8</td>   
</tr>
<tr>
<td>(2)</td>
<td>:heavy_check_mark:</td>
<td>:heavy_check_mark:</td>
<td></td>
<td></td>
<td>40.4</td>
<td>16M</td>
<td>20.0</td>
<td>44.9</td>
<td>38M</td>
<td>17.2</td>
<td>47.4</td>
<td>60M</td>
<td>12.1</td>   
</tr>
<tr>
<td>(3)</td>
<td>:heavy_check_mark:</td>
<td>:heavy_check_mark:</td>
<td>:heavy_check_mark:</td>
<td></td>
<td>41.8</td>
<td>16M</td>
<td>20.0</td>
<td>46.6</td>
<td>38M</td>
<td>17.2</td>
<td>49.2</td>
<td>60M</td>
<td>12.1</td>   
</tr>
<tr>
<td>(4)</td>
<td>:heavy_check_mark:</td>
<td>:heavy_check_mark:</td>
<td>:heavy_check_mark:</td>
<td>:heavy_check_mark:</td>
<td>41.6</td>
<td>13M</td>
<td>23.0</td>
<td>46.4</td>
<td>35M</td>
<td>19.5</td>
<td>49.1</td>
<td>58M</td>
<td>13.0</td>   
</tr>
</tbody>
</table>

### Requirements

This codebase has been developed with the setting used in [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR): <br>Linux, CUDA>=9.2, GCC>=5.4, Python>=3.7, PyTorch>=1.5.1, and torchvision>=0.6.1.

We recommend you to use Anaconda to create a conda environment:
```bash
conda create -n deformable_detr python=3.7 pip
conda activate deformable_detr
conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
```

#### Compiling CUDA operators for deformable attention
```bash
cd ./ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

#### Other requirements
```bash
pip install -r requirements.txt
```

### Training

We used the below commands to train ViDT models with a single node having 8 NVIDIA V100 GPUs.

<details>
<summary>Run this command to train the <code>ViDT (Swin-nano)</code> model in the paper :</summary>
<pre><code>
python -m torch.distributed.launch \
       --nproc_per_node=8 \
       --nnodes=1 \
       --use_env main.py \
       --method vidt \
       --backbone_name swin_nano \
       --epochs 50 \
       --lr 1e-4 \
       --min-lr 1e-7 \
       --batch_size 2 \
       --num_workers 2 \
       --aux_loss True \
       --with_box_refine True \
       --coco_path /path/to/coco \
       --output_dir /path/for/output
</code></pre>
</details>


<details>
<summary>Run this command to train the <code>ViDT (Swin-tiny)</code> model in the paper :</summary>
<pre><code>
python -m torch.distributed.launch \
       --nproc_per_node=8 \
       --nnodes=1 \
       --use_env main.py \
       --method vidt \
       --backbone_name swin_tiny \
       --epochs 50 \
       --lr 1e-4 \
       --min-lr 1e-7 \
       --batch_size 2 \
       --num_workers 2 \
       --aux_loss True \
       --with_box_refine True \
       --coco_path /path/to/coco \
       --output_dir /path/for/output
</code></pre>
</details>


<details>
<summary>Run this command to train the <code>ViDT (Swin-small)</code> model in the paper :</summary>
<pre><code>
python -m torch.distributed.launch \
       --nproc_per_node=8 \
       --nnodes=1 \
       --use_env main.py \
       --method vidt \
       --backbone_name swin_small \
       --epochs 50 \
       --lr 1e-4 \
       --min-lr 1e-7 \
       --batch_size 2 \
       --num_workers 2 \
       --aux_loss True \
       --with_box_refine True \
       --coco_path /path/to/coco \
       --output_dir /path/for/output
</code></pre>
</details>


<details>
<summary>Run this command to train the <code>ViDT (Swin-base)</code> model in the paper :</summary>
<pre><code>
python -m torch.distributed.launch \
       --nproc_per_node=8 \
       --nnodes=1 \
       --use_env main.py \
       --method vidt \
       --backbone_name swin_base_win7_22k \
       --epochs 50 \
       --lr 1e-4 \
       --min-lr 1e-7 \
       --batch_size 2 \
       --num_workers 2 \
       --aux_loss True \
       --with_box_refine True \
       --coco_path /path/to/coco \
       --output_dir /path/for/output
</code></pre>
</details>

When a large pre-trained ViDT model is available, distillation with token matching can be applied for training a smaller ViDT model.

<details>
<summary>Run this command when training ViDT (Swin-nano) using a large ViDT (Swin-base) via Knowledge Distillation :</summary>
<pre><code>
python -m torch.distributed.launch \
       --nproc_per_node=8 \
       --nnodes=1 \
       --use_env main.py \
       --method vidt \
       --backbone_name swin_nano \
       --epochs 50 \
       --lr 1e-4 \
       --min-lr 1e-7 \
       --batch_size 2 \
       --num_workers 2 \
       --aux_loss True \
       --with_box_refine True \
       --distil_model vidt_base \
       --distil_path /path/to/vidt_base (or url) \
       --coco_path /path/to/coco \
       --output_dir /path/for/output
</code></pre>
</details>

### Evaluation

<details>
<summary>Run this command to evaluate the <code>ViDT (Swin-nano)</code> model on COCO :</summary>
<pre><code>
python -m torch.distributed.launch \
       --nproc_per_node=8 \ 
       --nnodes=1 \
       --use_env main.py \
       --method vidt \
       --backbone_name swin_nano \
       --batch_size 2 \
       --num_workers 2 \
       --aux_loss True \
       --with_box_refine True \
       --coco_path /path/to/coco \
       --resume /path/to/vidt_nano \
       --pre_trained none \
       --eval True
</code></pre>
</details>

<details>
<summary>Run this command to evaluate the <code>ViDT (Swin-tiny)</code> model on COCO :</summary>
<pre><code>
python -m torch.distributed.launch \
       --nproc_per_node=8 \
       --nnodes=1 \
       --use_env main.py \
       --method vidt \
       --backbone_name swin_tiny \
       --batch_size 2 \
       --num_workers 2 \
       --aux_loss True \
       --with_box_refine True \
       --coco_path /path/to/coco \
       --resume /path/to/vidt_tiny\
       --pre_trained none \
       --eval True
</code></pre>
</details>

<details>
<summary>Run this command to evaluate the <code>ViDT (Swin-small)</code> model on COCO :</summary>
<pre><code>
python -m torch.distributed.launch \
       --nproc_per_node=8 \
       --nnodes=1 \
       --use_env main.py \
       --method vidt \
       --backbone_name swin_small \
       --batch_size 2 \
       --num_workers 2 \
       --aux_loss True \
       --with_box_refine True \
       --coco_path /path/to/coco \
       --resume /path/to/vidt_small \
       --pre_trained none \
       --eval True
</code></pre>
</details>

<details>
<summary>Run this command to evaluate the <code>ViDT (Swin-base)</code> model on COCO :</summary>
<pre><code>
python -m torch.distributed.launch \
       --nproc_per_node=8 \
       --nnodes=1 \
       --use_env main.py \
       --method vidt \
       --backbone_name swin_base_win7_22k \
       --batch_size 2 \
       --num_workers 2 \
       --aux_loss True \
       --with_box_refine True \
       --coco_path /path/to/coco \
       --resume /path/to/vidt_base \
       --pre_trained none \
       --eval True
</code></pre>
</details>

## Citation

Please consider citation if our paper is useful in your research.

```BibTeX
@article{song2021vidt,
  title={ViDT: An Efficient and Effective Fully Transformer-based Object Detector},
  author={Song, Hwanjun and Sun, Deqing and Chun, Sanghyuk and Jampani, Varun and Han, Dongyoon and Heo, Byeongho and Kim, Wonjae and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2110.03921},
  year={2021}
}
```

## License

```
Copyright 2021-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
