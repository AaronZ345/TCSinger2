[Back to TCSinger 2](../README.md)

# Train TCSinger 2

Run commands from the repository root in the environment described in the README. The example GPU IDs refer to existing devices; adjust them for your training setup.

## Data preparation 

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

## Training TCSinger 2

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
