# TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis

#### Yu Zhang, Wenxiang Guo, Changhao Pan, Dongyu Yao, Zhiyuan Zhu, Ziyue Jiang, Yuhan Wang, Tao Jin, Zhou Zhao | Zhejiang University

PyTorch implementation of **[TCSinger 2 (ACL 2025)](https://arxiv.org/abs/2505.14910): Customizable Multilingual Zero-shot Singing Voice Synthesis**.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2505.14910)
[![Demo](https://img.shields.io/badge/🚀%20Demo%20Page-blue)](https://aaronz345.github.io/TCSinger2Demo/) 
[![weixin](https://img.shields.io/badge/-WeChat@语音之家-000000?logo=wechat&logoColor=07C160)](https://mp.weixin.qq.com/s/Uvt27dt2S3TpBjJccl9mTA)
[![weixin](https://img.shields.io/badge/-WeChat@PaperWeekly-000000?logo=wechat&logoColor=07C160)](https://mp.weixin.qq.com/s/c6CPx14y_wASU_dSXKcAJQ)
[![zhihu](https://img.shields.io/badge/-知乎-000000?logo=zhihu&logoColor=0084FF)](https://zhuanlan.zhihu.com/p/1928562845236303816)
[![GitHub Stars](https://img.shields.io/github/stars/AaronZ345/TCSinger2?style=social&label=GitHub+Stars)](https://github.com/AaronZ345/TCSinger2)

Visit our [demo page](https://aaronz345.github.io/TCSinger2Demo/) for audio samples.

## News
- 2025.07: We released the code of TCSinger 2!
- 2025.07: We realeased the code of [STARS](https://github.com/gwx314/STARS)!
- 2025.05: TCSinger 2 is accepted by ACL 2025!

## Key Features
- We present **TCSinger 2**, a multi-task multilingual zero-shot SVS model with style transfer and style control based on various prompts.
- We introduce the **Blurred Boundary Content Encoder** for robust modeling and smooth transitions of phoneme and note boundaries.
- We design the **Custom Audio Encoder** using contrastive learning to extract styles from various prompts, while the **Flow-based Custom Transformer** with Cus-MOE and F0, enhances synthesis quality and style modeling.
- Experimental results show that TCSinger 2 outperforms baseline models in subjective and objective metrics across multiple tasks: **zero-shot style transfer, multi-level style control, cross-lingual style transfer, and speech-to-singing style transfer**.

## Quick Start
We provide an example of how you can train your own model and infer with TCSinger 2.

To try on your own dataset, clone this repo on your local machine with NVIDIA GPU + CUDA cuDNN and follow the instructions below.

### Dependencies

A suitable [conda](https://conda.io/) environment named `tcsinger2` can be created
and activated with:

```
conda create -n tcsinger2 python=3.10
conda install --yes --file requirements.txt
conda activate tcsinger2
```

### Multi-GPU

By default, this implementation uses as many GPUs in parallel as returned by `torch.cuda.device_count()`. 
You can specify which GPUs to use by setting the `CUDA_DEVICES_AVAILABLE` environment variable before running the training module.

## Train your own model

### Data Preparation 

1. Collect your own singing dataset, e.g., including [GTSinger](https://github.com/AaronZ345/GTSinger), and feel free to add extra data annotated with alignment tools, like [STARS](https://github.com/gwx314/STARS).  
2. Place `metadata.json` (fields: `ph`, `word`, `item_name`, `ph_durs`, `wav_fn`, `singer`, `ep_pitches`, `ep_notedurs`, `ep_types`, `emotion`, `singing_method`, `technique`) and `phone_set.json` (complete phoneme list) in the desired folder and update the paths in `preprocess/preprocess.py`.  (*A reference `metadata.json` is provided in **GTSinger***.) 
Please present the `singer` attribute as a description specifying the performer’s gender and vocal range, and render the `technique` attribute either as a concise text listing of skills or as a natural-language account that conveys their sequential order.
3. Extract F0 for each `.wav`, save as `*_f0.npy`, e.g. with **[RMVPE](https://github.com/Dream-High/RMVPE)**.
4. Download [HIFI-GAN](https://drive.google.com/drive/folders/1ve9cm_Yn3CQWSqkzMuRL33Uj1BNh51lR?usp=drive_link) as the vocoder in `useful_ckpts/hifigan` and [FLAN-T5](https://huggingface.co/google/flan-t5-large) in `useful_ckpts/flan-t5-large`.
5. Preprocess the dataset:

```bash
export PYTHONPATH=.
python preprocess/preprocess.py
```

*Tip: You may also convert your dataset directly to a `.csv` instead of using `metadata.json`.*

6. Compute mel-spectrograms:

```bash
python preprocess/mel_spec_48k.py --tsv_path data/new/data.tsv --num_gpus 1 --max_duration 20
```

7. Post-process:

```bash
python preprocess/postprocess.py
```

### Training TCSinger 2

1. Train the VAE module and duration predictor
```bash
python main.py --base configs/ae_singing.yaml -t --gpus 0,1,2,3,4,5,6,7
```

2. Train the main TCSinger 2 model
   
```bash
python main.py --base configs/tcsinger2.yaml -t --gpus 0,1,2,3,4,5,6,7
```

*Notes*  
- Adjust the compression ratio in the config files (and related scripts).  
- Change the padding length in the dataloader as needed.  
- To train the Custom Audio Encoder, format data as in `ldm/data/joinaudiodataset_con.py`, set the trained VAE path in `ae_con.yaml`, and proceed with training.

### Inference with TCSinger 2

```bash
python scripts/test_sing.py
```

*Replace the checkpoint path and CFG coefficient as required. For speech inputs, modify the VAE accordingly.*

Inference processes every eligible manifest item (duration up to 20 seconds) in manifest order by default, including a single item. To select a reproducible subset, pass `--sample-limit 10 --sample-seed 0`; a limit above the available item count selects all eligible items. The seed controls subset selection, not waveform generation.


## Acknowledgements

This implementation uses parts of the code from the following Github repos:
[Make-An-Audio-3](https://github.com/Text-to-Audio/Make-An-Audio-3),
[TCSinger](https://github.com/AaronZ345/TCSinger)
[Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X)
as described in our code.

## Citations ##

If you find this code useful in your research, please cite our work:
```bib
@article{zhang2025tcsinger,
  title={TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis},
  author={Zhang, Yu and Guo, Wenxiang and Pan, Changhao and Yao, Dongyu and Zhu, Zhiyuan and Jiang, Ziyue and Wang, Yuhan and Jin, Tao and Zhao, Zhou},
  journal={arXiv preprint arXiv:2505.14910},
  year={2025}
}
```

## Disclaimer ##

Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's singing without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

 ![visitors](https://visitor-badge.laobi.icu/badge?page_id=AaronZ345/TCSinger2)
