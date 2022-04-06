# An Extendable, Efficient and Effective Transformer-based Object Detector (Extension of VIDT published at ICLR2022)

**Please see the [vidt branch](https://github.com/naver-ai/vidt/tree/main) if you are interested in the vanilla ViDT model.**</br> **This is an extension of ViDT for joint-learning of object detection and instance segmentation.**

by [Hwanjun Song](https://scholar.google.com/citations?user=Ijzuc-8AAAAJ&hl=en&oi=ao)<sup>1</sup>, [Deqing Sun](https://scholar.google.com/citations?hl=en&user=t4rgICIAAAAJ)<sup>2</sup>, [Sanghyuk Chun](https://scholar.google.com/citations?user=4_uj0xcAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Varun Jampani](https://scholar.google.com/citations?hl=en&user=1Cv6Sf4AAAAJ)<sup>2</sup>, [Dongyoon Han](https://scholar.google.com/citations?user=jcP7m1QAAAAJ&hl=en)<sup>1</sup>, <br>[Byeongho Heo](https://scholar.google.com/citations?user=4_7rLDIAAAAJ&hl=en)<sup>1</sup>, [Wonjae Kim](https://scholar.google.com/citations?hl=en&user=UpZ41EwAAAAJ)<sup>1</sup>, and [Ming-Hsuan Yang](https://scholar.google.com/citations?hl=en&user=p9-ohHsAAAAJ)<sup>2,3</sup>

<sup>1</sup> NAVER AI Lab, <sup>2</sup> Google Research, <sup>3</sup> University California Merced

* **`April 6, 2022`:** **The official code is released!** </br> We obtained a light-weight transformer-based detector, achieving 47.0AP only with 14M parameters and 41.9 FPS (NVIDIA A100). </br> See [[C. Complete Analysis](#C)].

## ViDT+ for Joint-learning of Object Detection and Instance Segmentation

### Extension to ViDT+
<p align="center">
<img src="figures/overview.png " width="970"> 
</p>

We extend ViDT into ViDT+, supporting a joint-learning of object detection and instance segmentation in an end-to-end manner. Three new components have been leveraged for extensions: (1) *An efficient pyramid feature fusion (EPFF) module*, (2) *An unified query representation module*, and (3) two auxiliary losses of IoU-aware and token labeling.
Compared with the vanilla ViDT, ViDT+ provides a significant performance improvement without comprising inference speed. Only 1M parameters are added into the model.

### Evaluation

**Index:** [[A. ViT Backbone](#A)], [[B. Main Results](#B)], [[C. Complete Analysis](#C)]
```
|--- A. ViT Backbone used for ViDT
|--- B. Main Results in the ViDT+ Paper
     |--- B.1. VIDT+ compared with the vanilla ViDT for Object Detection
     |--- B.2. VIDT+ compared with other CNN-based methods for Object Detection and Instance Segmentation
|--- C. Complete Component Analysis
```

<a name="A"></a>
#### A. [ViT Backbone used for ViDT+](#content)

| Backbone and Size | Training Data | Epochs | Resulution | Params | ImageNet Acc. | Checkpoint |
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| `Swin-nano` | ImageNet-1K | 300 | 224 | 6M | 74.9% | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-swin/swin_nano_patch4_window7_224.pth) |
| `Swin-tiny` | ImageNet-1K | 300 | 224 | 28M | 81.2% | [Github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth) |
| `Swin-small` | ImageNet-1K | 300 | 224 | 50M | 83.2% | [Github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth) |
| `Swin-base` | ImageNet-22K | 90 | 224 | 88M | 86.3% | [Github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth) |

<a name="B"></a>
#### B. [Main Results in the ViDT+ Paper](#content)

`All the models were re-trained with the final version of source codes. Thus, the value may be very slightly different from those in the paper. Note that a single 'NVIDIA A100 GPU' was used to compute FPS for the input of batch size 1.`<br>
Compared with the vailla version, ViDT+ leverages three additional components or techniques:<br> 
(1) An efficient pyramid feature fusion (EPFF) module.<br>
(2) An unified query representation moudle (UQR).<br>
(3) Two additional losses of IoU-aware loss and token-labeling loss.

##### B.1. VIDT+ compared with the vanilla ViDT for Object Detection

| Method | Backbone | Epochs | AP | AP50 | AP75 | AP_S | AP_M | AP_L | Params | FPS | Checkpoint / Log |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | 
| ViDT+  | `Swin-nano` | 50 | 45.3 | 62.3 | 48.9 | 27.3 | 48.2 | 61.5 | 16M | 37.6 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-plus/vidt_plus_nano_det300.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-plus/vidt_plus_nano_det300.txt)|
| ViDT+   | `Swin-tiny` | 50 | 49.7 | 67.7 | 54.2 | 31.6 | 53.4 | 65.9 | 38M | 30.4 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-plus/vidt_plus_tiny_det300.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-plus/vidt_plus_tiny_det300.txt) |
| ViDT+   | `Swin-small` | 50 | 51.2 | 69.5 | 55.9 | 33.8 | 54.5 | 67.8 | 61M | 20.6 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-plus/vidt_plus_small_det300.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-plus/vidt_plus_small_det300.txt)|
| ViDT+   | `Swin-base` | 50 | 53.2 | 71.6 | 58.3 | 36.0 | 57.1 | 69.2 | 100M | 19.3 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-plus/vidt_plus_base_det300.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-plus/vidt_plus_base_det300.txt)|

| Method | Backbone | Epochs | AP | AP50 | AP75 | AP_S | AP_M | AP_L | Params | FPS | Checkpoint / Log |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | 
| ViDT    | `Swin-nano` | 50 | 40.4 | 59.9 | 43.0 | 23.1 | 42.8 | 55.9 | 15M | 40.8 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_nano_50.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_nano_50.txt)|
| ViDT    | `Swin-tiny` | 50 | 44.9 | 64.7 | 48.3 | 27.5 | 47.9 | 61.9 | 37M | 33.5 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_tiny_50.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_tiny_50.txt)|
| ViDT    | `Swin-small` | 50 | 47.4 | 67.7 | 51.2 | 30.4 | 50.7 | 64.6 | 60M | 24.7 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_small_50.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_small_50.txt)|
| ViDT    | `Swin-base` | 50 | 49.4 | 69.6 | 53.4 | 31.6 | 52.4 | 66.8 | 99M | 20.5 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_base_50.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_base_50.txt)|

##### B.2. VIDT+ compared with other CNN-based methods for Object Detection and Instance Segmentation

For fair comparison w.r.t the number of parameters, Swin-tiny and Swin-small backbones are used for ViDT+, which have similar number of parameters to ResNet-50 and ResNet-101, respectively. </br>
ViDT+ shows much higher detection AP than other joint-learning methods, but its segmentation AP is only higher than others for the medium- and large-size objects in general.

| Method | Backbone | Epochs | Box AP | Mask AP | Mask AP_S | Mask AP_M | Mask AP_L |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | 
| Mask R-CNN  | `ResNet-50 + FPN` | 36 |  41.3 |37.5 | 21.1 | 39.6 | 48.3 |
| HTC       | `ResNet-50 + FPN` | 36 |  44.9 |39.7 | 22.6 | 42.2 | 50.6 |
| SOLOv2    | `ResNet-50 + FPN` | 72 |40.4 |  38.8 | 16.5 | 41.7 | 56.2 | 
| QueryInst | `ResNet-50 + FPN` | 36 |45.6 | 40.6 | 23.4 | 42.5 | 52.8 | 
| SOLQ      | `ResNet-50`       | 50 | 47.8 | 39.7 | 21.5 | 42.5 | 53.1 | 
| **ViDT+**  | `Swin-tiny` | 50 | 49.7 | 39.5 | 21.5 | 43.4 | 58.2 |

| Method | Backbone | Epochs | Box AP  | Mask AP | Mask AP_S | Mask AP_M | Mask AP_L | 
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | 
| Mask R-CNN  | `ResNet-101 + FPN` | 50 | 41.3 |  38.8 | 21.8 | 41.4 | 50.5 |
| HTC       | `ResNet-101 + FPN` | 50 | 44.3 | 40.8 | 23.0 | 43.5 | 58.2 | 
| SOLOv2    | `ResNet-101 + FPN` | 50 |42.6 |  39.7 | 17.3 | 42.9 | 58.2 | 
| QueryInst | `ResNet-101 + FPN` | 50 | 48.1 |  42.8 | 24.6 | 45.0 | 58.2 | 
| SOLQ      | `ResNet-101`       | 50 |48.7 |  40.9 | 22.5 | 43.8 | 58.2 | 
| **ViDT+**  | `Swin-small`      | 50 | 51.2 |  40.8 | 22.6 | 44.3 | 60.1 |


<a name="C"></a>
#### C. [Complete Component Analysis](#content)

We combined the four proposed components (even with distillation with token matching and decoding layer drop) to achieve high accuracy and speed for object detection. For distillation, ViDT (Swin-base) trained for 50 epochs was used for all models.

We combined all the proposed components (even with longer training epochs and decoding layer dropping) to achive high accuracy and speed for object detection. As summarized in below table, there are *eight* components for extension: (1) RAM, (2) the neck decoder, (3) the IoU-aware and token labeling losses, (4) the EPFF module, (5) the UQR module, (6) the use of more detection tokens, (6) the use of longer training epochs, and (8) decoding layer drop.

`The numbers (2), (6), and (8) are the performance of the vanilla ViDT, its extension to ViDT+, and the fully optimized ViDT+.`

<table>
<thead>
<tr>
<th colspan="1"> </th>
<th colspan="1">Added</th>
<th colspan="3">Swin-nano</th>
<th colspan="3">Swin-tiny</th>
<th colspan="3">Swin-small</th>
</tr>
<tr>
<th>#</th>
<th>Module</th>
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
<td>+ RAM</td>
<td>28.7</td>
<td>7M</td>
<td>72.4</td>
<td>36.3</td>
<td>29M</td>
<td>51.8</td>
<td>41.6</td>
<td>52M</td>
<td>33.5</td>   
</tr>
<tr>
<td>(2)</td>
<td>+ Encoder-free Neck</td>
<td>40.4</td>
<td>15M</td>
<td>40.8</td>
<td>44.8</td>
<td>37M</td>
<td>33.5</td>
<td>47.5</td>
<td>60M</td>
<td>24.7</td>   
</tr>
<tr>
<td>(3)</td>
<td>+ IoU-aware & Token Label</td>
<td>41.0</td>
<td>15M</td>
<td>40.8</td>
<td>45.9</td>
<td>37M</td>
<td>33.5</td>
<td>48.5</td>
<td>60M</td>
<td>24.7</td>   
</tr>
<tr>
<td>(4)</td>
<td>+ EPFF Module</td>
<td>42.5</td>
<td>16M</td>
<td>38.0</td>
<td>47.1</td>
<td>38M</td>
<td>30.9</td>
<td>49.3</td>
<td>61M</td>
<td>23.0</td>   
</tr>
<tr>
<td>(5)</td>
<td>+ UQR Module</td>
<td>43.9</td>
<td>16M</td>
<td>38.0</td>
<td>47.9</td>
<td>38M</td>
<td>30.9</td>
<td>50.1</td>
<td>61M</td>
<td>23.0</td>   
</tr>
<tr>
<td>(6)</td>
<td>+ 300 [DET] Tokens</td>
<td>45.3</td>
<td>16M</td>
<td>37.6</td>
<td>49.7</td>
<td>38M</td>
<td>30.4</td>
<td>51.2</td>
<td>61M</td>
<td>22.6</td>   
</tr>
<tr>
<td>(7)</td>
<td>+ 150 Training Epochs</td>
<td>47.6</td>
<td>16M</td>
<td>37.6</td>
<td>51.4</td>
<td>38M</td>
<td>30.4</td>
<td>52.3</td>
<td>61M</td>
<td>22.6</td>   
</tr>
<tr>
<td>(8)</td>
<td>+ Decoding Layer Drop</td>
<td>47.0</td>
<td>14M</td>
<td>41.9</td>
<td>50.8</td>
<td>36M</td>
<td>33.9</td>
<td>51.8</td>
<td>59M</td>
<td>24.6</td>   
</tr>
</tbody>
</table>

The optimized ViDT+ models can be found:</br>
[ViDT+ (Swin-nano)](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-plus-optimized/vidt_plus_swin_nano_optimized.pth), [ViDT+ (Swin-tiny)](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-plus-optimized/vidt_plus_swin_tiny_optimized.pth), and [ViDT+ (Swin-small)](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt-plus-optimized/vidt_plus_swin_small_optimized.pth).

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
## Training and Evaluation

If you want to test with a single GPU, see [colab examples](https://github.com/EherSenaw/ViDT_colab/blob/main/vidt_colab.ipynb). Thanks to [EherSenaw](https://github.com/EherSenaw) for making this example.<br>
The below codes are for training with multi GPUs.

### Training for ViDT+

We used the below commands to train ViDT+ models with a single node having 8 NVIDIA GPUs.

<details>
<summary>Run this command to train the <code>ViDT+ (Swin-nano)</code> model in the paper :</summary>
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
       --det_token_num 300 \
       --epff True \
       --token_label True \
       --iou_aware True \
       --with_vector True \
       --masks True \
       --coco_path /path/to/coco \
       --output_dir /path/for/output
</code></pre>
</details>


<details>
<summary>Run this command to train the <code>ViDT+ (Swin-tiny)</code> model in the paper :</summary>
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
       --det_token_num 300 \
       --epff True \
       --token_label True \
       --iou_aware True \
       --with_vector True \
       --masks True \
       --coco_path /path/to/coco \
       --output_dir /path/for/output
</code></pre>
</details>


<details>
<summary>Run this command to train the <code>ViDT+ (Swin-small)</code> model in the paper :</summary>
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
       --det_token_num 300 \
       --epff True \
       --token_label True \
       --iou_aware True \
       --with_vector True \
       --masks True \
       --coco_path /path/to/coco \
       --output_dir /path/for/output
</code></pre>
</details>


<details>
<summary>Run this command to train the <code>ViDT+ (Swin-base)</code> model in the paper :</summary>
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
       --det_token_num 300 \
       --epff True \
       --token_label True \
       --iou_aware True \
       --with_vector True \
       --masks True \
       --coco_path /path/to/coco \
       --output_dir /path/for/output
</code></pre>
</details>

### Evaluation for ViDT+

<details>
<summary>Run this command to evaluate the <code>ViDT+ (Swin-nano)</code> model on COCO :</summary>
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
       --det_token_num 300 \
       --epff True \
       --coco_path /path/to/coco \
       --resume /path/to/vidt_nano \
       --pre_trained none \
       --eval True
</code></pre>
</details>

<details>
<summary>Run this command to evaluate the <code>ViDT+ (Swin-tiny)</code> model on COCO :</summary>
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
       --det_token_num 300 \
       --epff True \
       --coco_path /path/to/coco \
       --resume /path/to/vidt_tiny\
       --pre_trained none \
       --eval True
</code></pre>
</details>

<details>
<summary>Run this command to evaluate the <code>ViDT+ (Swin-small)</code> model on COCO :</summary>
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
       --det_token_num 300 \
       --epff True \
       --coco_path /path/to/coco \
       --resume /path/to/vidt_small \
       --pre_trained none \
       --eval True
</code></pre>
</details>

<details>
<summary>Run this command to evaluate the <code>ViDT+ (Swin-base)</code> model on COCO :</summary>
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
       --det_token_num 300 \
       --epff True \
       --coco_path /path/to/coco \
       --resume /path/to/vidt_base \
       --pre_trained none \
       --eval True
</code></pre>
</details>

### Training for ViDT

We used the below commands to train ViDT models with a single node having 8 NVIDIA GPUs.

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
       --det_token_num 100 \
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
       --det_token_num 100 \
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
       --det_token_num 100 \
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
       --det_token_num 100 \
       --coco_path /path/to/coco \
       --output_dir /path/for/output
</code></pre>
</details>

### Evaluation for ViDT

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
       --det_token_num 100 \
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
       --det_token_num 100 \
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
       --det_token_num 100 \
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
       --det_token_num 100 \
       --coco_path /path/to/coco \
       --resume /path/to/vidt_base \
       --pre_trained none \
       --eval True
</code></pre>
</details>

## Citation

Please consider citation if our paper is useful in your research.

```BibTeX
@inproceedings{song2022vidt,
  title={ViDT: An Efficient and Effective Fully Transformer-based Object Detector},
  author={Song, Hwanjun and Sun, Deqing and Chun, Sanghyuk and Jampani, Varun and Han, Dongyoon and Heo, Byeongho and Kim, Wonjae and Yang, Ming-Hsuan},
  booktitle={International Conference on Learning Representation},
  year={2022}
}
```

```BibTeX
@inproceedings{song2022vidtplus,
  title={An Extendable, Efficient and Effective Transformer-based Object Detector},
  author={Song, Hwanjun and Sun, Deqing and Chun, Sanghyuk and Jampani, Varun and Han, Dongyoon and Heo, Byeongho and Kim, Wonjae and Yang, Ming-Hsuan},
  year={2022}
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
