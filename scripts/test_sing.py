import os
from pathlib import Path
import csv
import argparse
import traceback
import sys
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.distributed import init_process_group
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
import soundfile as sf
import matplotlib.pyplot as plt
from vocoder.hifigan import HifiGAN
from ldm.models.diffusion.cfm1_audio_sampler import CFMSampler
from ldm.util import instantiate_from_config
import random
import pandas as pd
from typing import Union
import ast
import torch.nn.functional as F 


language_set = ['Chinese','English','French','German','Italian','Japanese','Korean','Russian','Spanish']

ph_set = [
    "tʰ_ko",
    "ɔ_it",
    "d_it",
    "dʑ_ja",
    "ɣ_es",
    "in",
    "ʒ_fr",
    "tʲ_ko",
    "p_it",
    "ɕ͈_ko",
    "t͡ʃː_it",
    "UW1",
    "dʲ_ko",
    "IH2",
    "NG",
    "ZH",
    "ɡ_es",
    "ɾ_ja",
    "ɛ_ko",
    "IH1",
    "j",
    "ʎ_fr",
    "ɨː_ja",
    "kː_it",
    "a",
    "OW1",
    "f",
    "AO0",
    "ou",
    "k_ja",
    "l_it",
    "o_ja",
    "F",
    "ʂː_ru",
    "AW1",
    "i_ko",
    "ɰ_ko",
    "tɕ_ru",
    "uai",
    "z_it",
    "ie",
    "u_ko",
    "mː_it",
    "ʃ_it",
    "j_fr",
    "ch",
    "n_es",
    "ɛ_de",
    "ɴ_ja",
    "tɕʷ_ko",
    "b_it",
    "iou",
    "IY1",
    "ɛ_fr",
    "k͈ʷ_ko",
    "s̪_ru",
    "b_ja",
    "aj_de",
    "c_ru",
    "e_it",
    "pf_de",
    "ç_es",
    "<SP>",
    "v_it",
    "ŋ_fr",
    "ʔ_ja",
    "S",
    "tː_ja",
    "ɑ_fr",
    "x_es",
    "r_es",
    "sh",
    "ɡ_fr",
    "ʌː_ko",
    "ɟ_es",
    "zʲ_ru",
    "AW0",
    "AA0",
    "Y",
    "ɟ_de",
    "e_ru",
    "ə_ru",
    "P",
    "t_de",
    "f_ru",
    "l_fr",
    "k_ko",
    "ɐ_ru",
    "t͡s_it",
    "UH0",
    "s_ja",
    "ʃ_fr",
    "β_es",
    "n_ja",
    "AY2",
    "ʌ_ko",
    "p_ko",
    "dʲ_ru",
    "ŋ_de",
    "iang",
    "b_ru",
    "o",
    "AA2",
    "a_ru",
    "t",
    "o_ru",
    "x_ru",
    "ɛ̃_fr",
    "oː_ko",
    "AE1",
    "v",
    "D",
    "z̪_ru",
    "b_es",
    "UW2",
    "uo",
    "EY1",
    "œ_de",
    "øː_de",
    "yː_de",
    "CH",
    "o_fr",
    "bʲ_ja",
    "u",
    "sː_it",
    "<AP>",
    "w_es",
    "UH2",
    "ø_fr",
    "ʎ_es",
    "ɪ_de",
    "er",
    "k_de",
    "ç_ru",
    "L",
    "ʐ_ru",
    "G",
    "ʑ_ja",
    "sʲ_ru",
    "t_fr",
    "dː_it",
    "ɥ_fr",
    "s_fr",
    "x_de",
    "ɨ_ru",
    "EY2",
    "kʰ_de",
    "ə_fr",
    "l",
    "ð_es",
    "n̪ː_ru",
    "a_it",
    "ɕː_ru",
    "z_de",
    "w_ja",
    "ɲ_ja",
    "r_ru",
    "TH",
    "rː_ru",
    "t͈_ko",
    "j_ja",
    "t_ko",
    "w_fr",
    "AW2",
    "uː_ko",
    "t̪ː_ru",
    "c_ko",
    "OY1",
    "e_es",
    "ʁ_fr",
    "ŋ_ko",
    "eː_ko",
    "ɔ̃_fr",
    "fʲ_ru",
    "j_es",
    "ʃ_de",
    "mː_ko",
    "d",
    "ɲ_ko",
    "OW0",
    "j_it",
    "AY1",
    "h_ja",
    "dz_ja",
    "f_es",
    "AA1",
    "ʎ_ru",
    "k_fr",
    "cʰ_de",
    "ɲ_fr",
    "c͈_ko",
    "θ_es",
    "pː_it",
    "d_ko",
    "pʰ_de",
    "e_ko",
    "d̪_es",
    "EH0",
    "m_es",
    "ɲ_de",
    "UW0",
    "ɫ_ru",
    "ŋ_es",
    "ɔ_de",
    "M",
    "B",
    "tʃ_fr",
    "V",
    "pʲ_ko",
    "IY0",
    "eː_ja",
    "aː_ja",
    "sʰ_ko",
    "ɨ̥_ja",
    "z_fr",
    "d_de",
    "m_fr",
    "s͈_ko",
    "uang",
    "k",
    "eː_de",
    "hː_it",
    "ɵ_ru",
    "m_it",
    "ʏ_de",
    "f_it",
    "ɕ_ja",
    "ɰ̃_ja",
    "uei",
    "i",
    "ei",
    "ɴː_ja",
    "tɕ_ko",
    "ʊ_ru",
    "OY2",
    "i_es",
    "IY2",
    "ian",
    "u_fr",
    "pʷ_ko",
    "ɦ_ko",
    "j_ko",
    "rʲ_ru",
    "c_es",
    "iao",
    "en",
    "t_ja",
    "y_fr",
    "ə_de",
    "tɕ͈_ko",
    "AE0",
    "tɕ_ja",
    "d̪_ru",
    "a_de",
    "ER1",
    "tʃ_es",
    "ɪ_ru",
    "ɸʷ_ko",
    "ER2",
    "ç_de",
    "tɕʰ_ko",
    "ɥ_ko",
    "T",
    "EH1",
    "d_ja",
    "ɲ_it",
    "DH",
    "eng",
    "ç_ko",
    "tː_it",
    "i_it",
    "ɲː_ru",
    "h",
    "o_it",
    "vʲ_ru",
    "sʷ_ko",
    "AH0",
    "oː_ja",
    "ç_ja",
    "ɕʰ_ko",
    "uː_de",
    "ɸ_ja",
    "q",
    "fː_it",
    "m_ko",
    "ɐ_de",
    "k̚_ko",
    "g",
    "x_ko",
    "s_ko",
    "ɕ_ru",
    "tʃ_de",
    "i_ja",
    "N",
    "JH",
    "IH0",
    "t̪s̪ː_ru",
    "dʑ_ko",
    "n̩_de",
    "n_de",
    "t̚_ko",
    "AH1",
    "t͡ʃ_it",
    "zh",
    "ʐː_ru",
    "o_es",
    "c_fr",
    "ʁ_de",
    "vn",
    "n_fr",
    "ɯ̥_ja",
    "ɟ_ja",
    "r_it",
    "an",
    "lː_it",
    "ang",
    "a_fr",
    "ɡ_de",
    "r",
    "nː_ja",
    "ɐ_ko",
    "k_ru",
    "R",
    "u_it",
    "AE2",
    "f_fr",
    "rː_it",
    "ɾʲ_ja",
    "p_ja",
    "bʲ_ru",
    "tʷ_ko",
    "mʲ_ja",
    "æ_ru",
    "e_fr",
    "m_de",
    "m_ru",
    "n",
    "ɟʝ_es",
    "t̪_es",
    "ia",
    "n_it",
    "iong",
    "ɡ_it",
    "u_ru",
    "v_ru",
    "ɲ_ru",
    "ɔʏ_de",
    "tʰ_de",
    "mʲ_ru",
    "pʰ_ko",
    "ʎ_it",
    "t̪s̪_ru",
    "HH",
    "m̩_de",
    "a_ja",
    "ɛː_ko",
    "OW2",
    "uen",
    "t̪_ru",
    "e_ja",
    "p_de",
    "i_ru",
    "bʲ_ko",
    "j_de",
    "AH2",
    "k_it",
    "ɯ_ja",
    "oː_de",
    "EY0",
    "d͡ʒ_it",
    "ɨ_ja",
    "aː_de",
    "p̚_ko",
    "s_it",
    "ɛ_it",
    "ai",
    "l_es",
    "k_es",
    "K",
    "ʉ_ru",
    "o_ko",
    "EH2",
    "v_de",
    "ɾ_ko",
    "s",
    "p_ru",
    "UH1",
    "ts_ja",
    "i̥_ja",
    "ts_de",
    "ʂ_ru",
    "z",
    "b_de",
    "pʲ_ru",
    "Z",
    "v_fr",
    "aw_de",
    "k͈_ko",
    "f_de",
    "n̪_ru",
    "œ_fr",
    "ɟ_fr",
    "SH",
    "d͡ʒː_it",
    "ɾʲ_ko",
    "ua",
    "mʲ_ko",
    "ɨː_ko",
    "h_de",
    "pː_ja",
    "c",
    "s_de",
    "l_de",
    "m",
    "j_ru",
    "cʰ_ko",
    "ɾ_es",
    "p",
    "w_ko",
    "p_es",
    "ER0",
    "ɨ_ko",
    "z_ja",
    "b_fr",
    "s̪ː_ru",
    "ɡ_ja",
    "s_es",
    "ʝ_es",
    "c_ja",
    "h_ko",
    "d_fr",
    "kʷ_ko",
    "ɲ_es",
    "iː_ja",
    "uan",
    "van",
    "ao",
    "AO1",
    "ve",
    "e",
    "mʲ_fr",
    "t͈ʲ_ko",
    "ɡ_ru",
    "ʊ_de",
    "AY0",
    "n_ko",
    "ing",
    "kʰ_ko",
    "W",
    "iː_de",
    "nː_it",
    "p͈_ko",
    "ɟ_ru",
    "a_es",
    "x",
    "ɔ_fr",
    "m_ja",
    "i_fr",
    "ɛ_ru",
    "ɑ̃_fr",
    "ong",
    "bː_it",
    "t_it",
    "AO2",
    "tsʲ_ru",
    "u_es",
    "p_fr",
    "iː_ko",
    "b",
    "ɭ_ko"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='configs/tcsinger2.yaml'
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="logs/2025-07-10T11-26-43_tcsinger2/checkpoints/last.ckpt"
    )
    parser.add_argument(
        "--manifest_path",  # 训练时候的 tsv，用来直接构造 infer 时的 dataset
        type=str,
        default='data/sing/new.tsv'
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,  # if it's 1, only condition is taken into consideration
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default='1-4',  # use '-' to separate, such as '1-2-3-4'
        # default=None,  # use '-' to separate
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='test'
    )
    parser.add_argument(
        "--save_plot",
        action='store_true'
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=48000
    )
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="Optional positive limit on eligible input items; default: all.")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="Seed for selecting a subset when --sample-limit is used.")
    return parser.parse_args()

def save_df_to_tsv(dataframe, path: Union[str, Path]):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

def safe_path(path):
    os.makedirs(Path(path).parent, exist_ok=True)
    return path

def load_samples_from_tsv(tsv_path):
    tsv_path = Path(tsv_path)
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {tsv_path}")
    with open(tsv_path) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        samples = [dict(e) for e in reader]
    if len(samples) == 0:
        print(f"warning: empty manifest: {tsv_path}")
        return []
    return samples

def load_dict_from_tsv(tsv_path, key):
    samples = load_samples_from_tsv(tsv_path)
    samples = {sample[key]: sample for sample in samples}
    return samples

def initialize_model(config, ckpt, device='cpu'):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt, map_location='cpu')["state_dict"], strict=False)

    model = model.to(device)
    model.cond_stage_model.to(model.device)
    model.cond_stage_model.device = model.device
    print(model.device, device, model.cond_stage_model.device)
    # sampler = DDIMSampler(model)
    sampler = CFMSampler(model,num_timesteps=1000)

    return sampler

def save_mel(spec, save_path):
    fig = plt.figure(figsize=(14, 10))
    heatmap = plt.pcolor(spec, vmin=-6, vmax=1.5)
    fig.colorbar(heatmap)
    fig.savefig(save_path, format='png')
    plt.close(fig)

def handle_exception(err, skipped_name=''):
    _, exc_value, exc_tb = sys.exc_info()
    tb = traceback.extract_tb(exc_tb)[-1]
    if skipped_name != '':
        print(f'skip {skipped_name}, {err}: {exc_value} in {tb[0]}:{tb[1]} "{tb[2]}" in {tb[3]}')
    else:
        print(f'{err}: {exc_value} in {tb[0]}:{tb[1]} "{tb[2]}" in {tb[3]}')


class InferDataset(Dataset):
    def __init__(self, manifest_path, sample_limit=None, sample_seed=0):
        super().__init__()
        # Load all samples from the manifest
        samples = load_samples_from_tsv(manifest_path)
        self.items_dict = {sample['item_name']: sample for sample in samples}

        self.mel_num = 80
        self.mel_downsample_rate = 2    # temporal downsample rate
        self.min_factor = 8
        self.min_batch_len = 376
        self.pad_value = -5
        self.max_batch_len = 3760
        
        self.pred_list = [name for name, item in self.items_dict.items()
                          if float(item['duration']) <= 20]
        if not self.pred_list:
            raise ValueError("Manifest has no eligible items with duration <= 20 seconds")
        if sample_limit is not None:
            if sample_limit < 1:
                raise ValueError("sample_limit must be positive")
            self.pred_list = random.Random(sample_seed).sample(
                self.pred_list, min(sample_limit, len(self.pred_list)))

        self.items = []
        for item_name in self.pred_list:
            item = self.items_dict[item_name]
            item['name'] = item_name
            self.items.append(item)
        self.language_to_id = {language: idx for idx, language in enumerate(language_set)}

        del self.items_dict

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        data = self.items[index]

        item = {}
        start = -1
        valid_spec = True
        # try:
        spec = np.load(data['mel_path'])  # mel spec [80, T]
        spec_len = spec.shape[1]
        
        # 随机选择
        if spec.shape[1] > self.max_batch_len:
            print('too long', spec.shape[1])
            start=0
            spec = spec[:, start: start + self.max_batch_len]
            spec_len = self.max_batch_len

        prompt_list = ast.literal_eval(data['prompt_mel'])
        prompt_mel = random.choice(prompt_list)
        prompt=np.load(prompt_mel) 
        if prompt.shape[1] > self.max_batch_len:
            prompt = prompt[:, :self.max_batch_len]

        # 读取f0
        f0_path = data['audio_path'].replace('.wav', '_f0.npy')
        f0 = np.load(f0_path)  # [T,]
        if f0.shape[0] > spec_len:
            start = 0
            f0 = f0[start:start+spec_len]
        else:
            # 如果f0比mel短则截断mel或稍后pad
            min_len = min(f0.shape[0], spec_len)
            f0 = f0[:min_len]
            spec = spec[:, :min_len]
            spec_len = min_len
        f0 = f0[np.newaxis, :]  # [1, T]

        ph = ast.literal_eval(data['ph'])
        # if you want to use pred dur, you can use the pred_duration function of the first_stage_model, this is just a simple test
        ph_durs = ast.literal_eval(data['ph_durs'])
        ep_pitches = ast.literal_eval(data['ep_pitches'])
        ep_notedurs = ast.literal_eval(data['ep_notedurs'])
        ep_types = ast.literal_eval(data['ep_types'])

        assert len(ph) == len(ph_durs) == len(ep_pitches) == len(ep_notedurs) == len(ep_types),f'item_name: {data["item_name"]}, ph: {ph}, ph_durs: {ph_durs}, ep_pitches: {ep_pitches}, ep_notedurs: {ep_notedurs}, ep_types: {ep_types}, tech: {tech}'
        
        # mel2ph
        audio_sample_rate = 48000
        hop_size = 256

        mel2ph = np.zeros((spec_len,), dtype=np.int32)
        current_frame = 0
        for i_ph, dur in enumerate(ph_durs):
            if dur <= 0:
                # 如跳过：继续下一个音素
                # 或者将dur视作0，不填充
                continue
            # 持续时间转为mel帧数
            frames = int(dur * audio_sample_rate / hop_size + 0.5)
            end_frame = current_frame + frames
            if end_frame > spec_len:
                # 如果超过mel长度，则截断
                end_frame = spec_len
            # 将对应区间的mel帧标记为此音素
            # 通常mel2ph从1开始编号音素
            mel2ph[current_frame:end_frame] = i_ph + 1
            # 更新下一个音素的起始帧位置
            current_frame = end_frame
            # 如果已经填满整个mel长度，则跳出循环
            if current_frame >= spec_len:
                break

        ph2id = {ph: idx for idx, ph in enumerate(ph_set)}
        ph = [ph2id[p] for p in ph]

        language_id = self.language_to_id[data['language']]

        caption=''
        caption += f', A Singer: {data["singer"]}'
        caption += f', in Language: {data["language"]},'
        if data['emotion'] !='no':
            caption += f', with Emotion: {data["emotion"]} and Singing Method: {data["singing_method"]},'
        if data['technique'] !='no':
            caption += f', using Technique: {data["technique"]}'

        return {
            'audio_path': data['audio_path'],
            'prompt'    : torch.FloatTensor(prompt),
            'image'     : torch.FloatTensor(spec),   # 依旧保留，便于 debug
            'caption'   : caption,
            'f0'        : torch.FloatTensor(f0),
            'name'      : data['item_name'],
            'ph_seq'    : torch.LongTensor(ph),       
            'ep_pitch_seq': torch.LongTensor(ep_pitches),  # [N_ph]
            'ep_notedurs': torch.FloatTensor(ep_notedurs),  # [N_ph]
            'ep_types'  : torch.LongTensor(ep_types),     # [N_ph
            'language'  : torch.LongTensor([language_id]),
            'ph_durs': torch.FloatTensor(ph_durs),  # [N_ph]
        }     
        # return item

    def collator(self, samples):

        return samples

def normalize_loudness(wav, target_loudness):
    rms = np.sqrt(np.mean(wav ** 2))
    loudness = 20 * np.log10(rms)
    gain = target_loudness - loudness
    normalized_wav = wav * 10 ** (gain / 20)
    return normalized_wav

@torch.no_grad()
def gen_song(rank, args):
    if args.num_gpus > 1:
        init_process_group(backend=args.dist_config['dist_backend'], init_method=args.dist_config['dist_url'],
                           world_size=args.dist_config['world_size'] * args.num_gpus, rank=rank)

    dataset = InferDataset(args.manifest_path, args.sample_limit, args.sample_seed)
    ds_sampler = DistributedSampler(dataset, shuffle=False) if args.num_gpus > 1 else None
    loader = DataLoader(dataset, sampler=ds_sampler, collate_fn=dataset.collator, batch_size=1, num_workers=40, drop_last=False)

    device = torch.device(f"cuda:{int(rank)}")
    sampler = initialize_model(args.config, args.ckpt, device)
    vocoder_cfg = {
        'target': 'vocoder.hifigan.hifigan_nsf.HifiGAN_NSF',
        'params': {
            'vocoder_ckpt': 'useful_ckpts/hifigan',
            'use_nsf': True
        }
    }
    vocoder = instantiate_from_config(vocoder_cfg)

    if args.scales != '' or args.scales is not None:
        scales = [float(s) for s in args.scales.split('-')]
    else:
        scales = [args.scale]

    save_dir = args.save_dir
    loader = tqdm(loader) if rank == 0 else loader
    item_idx = 0
    results = []
    csv_data = {}
    csv_data['audio_path']=[]
    csv_data['caption']=[]
    csv_data['name']=[]
    
    for batch in loader:
        item = batch[0]
        item_name = item['name']

        f0 = item['f0'].to(device)
        ph_seq        = item['ph_seq'].to(device)        # [N_ph]
        ep_pitch_seq  = item['ep_pitch_seq'].to(device)  # [N_ph]
        ep_notedurs = item['ep_notedurs'].to(device)  # [N_ph]
        ep_types    = item['ep_types'].to(device)     # [N_ph]
        caption = item['caption']
        prompt = item['prompt'].to(device)
        language = item['language'].to(device)
        ph_durs = item['ph_durs'].to(device)  # [N_ph]
        
        uncond_caption = ""

        cond_gtcodec_wavs_dict = {}

        with torch.no_grad():
            # pred_duration: (B, N_ph) → log 时长
            padding_mask = (ph_seq == 475)
            log_dur = sampler.model.first_stage_model.pred_duration(
                ph_seq.unsqueeze(0), ep_notedurs.unsqueeze(0), ep_pitch_seq.unsqueeze(0), ep_types.unsqueeze(0),padding_mask.unsqueeze(0),
            )[0]                    

        hop = 256
        sr  = 48000
        dur_frames = (torch.exp(log_dur) - 1)*sr/hop
        dur_frames = dur_frames.round().clamp(min=1).long()

        # ---------- Step 3: 展开到帧级 ----------
        mel2ph = torch.repeat_interleave(
            torch.arange(1, len(dur_frames) + 1, device=device), dur_frames
        )                                   # [T]

        # 同步展开 ph / midi：
        ph_array        = torch.repeat_interleave(ph_seq,       dur_frames)  # [T]
        ep_pitch_array  = torch.repeat_interleave(ep_pitch_seq, dur_frames)  # [T]

        # ---------- Step 4: (可选) 如果生成长度和 spec 不一致，就截断或 pad ----------
        spec_len = ph_array.shape[0]
        if f0.shape[-1] < spec_len:             # pad f0
            pad = spec_len - f0.shape[-1]
            f0 = F.pad(f0, (0, pad), value=0)
        elif f0.shape[-1] > spec_len:           # 截断 f0
            f0 = f0[..., :spec_len]

        # ---------- Step 5: 生成 mask ----------
        ph_val, start_idx, length = mel2ph[0], 0, 1
        for i in range(1, spec_len):
            if mel2ph[i] == ph_val and ph_val != 0:
                length += 1
            else:
                if length > 16:
                    seg_start = torch.randint(start_idx, start_idx + length - 15, (1,)).item()
                    seg_end   = seg_start + 16
                    for rng in [range(start_idx, seg_start), range(seg_end, start_idx + length)]:
                        mel2ph[rng]       = 0
                        ph_array[rng]     = 475
                        ep_pitch_array[rng] = 99
                ph_val, start_idx, length = mel2ph[i], i, 1
        if length > 16:
            seg_start = torch.randint(start_idx, start_idx + length - 15, (1,)).item()
            seg_end   = seg_start + 16
            for rng in [range(start_idx, seg_start), range(seg_end, start_idx + length)]:
                mel2ph[rng]       = 0
                ph_array[rng]     = 475
                ep_pitch_array[rng] = 99

        # ---------- Step 6: 整回 batch，用原字段名，后续代码基本不用动 ----------
        item['ph']        = ph= ph_array.unsqueeze(0).long()        # [1, T]
        item['ep_pitches'] = ep_pitches=ep_pitch_array.unsqueeze(0).long() # [1, T]
        item['mel2ph']    = mel2ph=mel2ph.unsqueeze(0).long()          # [1, T]

        for scale in scales:
            latent_length = int(mel2ph.shape[1]//4)
            start_code = torch.randn(args.n_samples, sampler.model.first_stage_model.embed_dim, latent_length).to(device=device, dtype=torch.float32)
            shape = [sampler.model.first_stage_model.embed_dim, latent_length]
            condition = {
                'caption': [caption] * args.n_samples,
                'acoustic': {'f0':torch.stack([f0] * args.n_samples),'midi':torch.stack([ep_pitches] * args.n_samples).long(),'ph':torch.stack([ph] * args.n_samples).long(),'mel2ph':torch.stack([mel2ph] * args.n_samples).long(),'prompt':torch.stack([prompt] * args.n_samples),'language':language},
                'name': [item_name] * args.n_samples,
                'infer':True
            }
            c = sampler.model.get_learned_conditioning(condition)
            uc = None
            if args.scale != 1.0:
                uncondition = {
                    'caption': [uncond_caption] * args.n_samples,
                    'acoustic': {'f0':torch.stack([f0] * args.n_samples),'midi':torch.stack([ep_pitches] * args.n_samples).long(),'ph':torch.stack([ph] * args.n_samples).long(),'mel2ph':torch.stack([mel2ph] * args.n_samples).long(),'prompt':torch.stack([prompt] * args.n_samples),'language':language},
                    'name': [item_name] * args.n_samples,
                    'infer':True
                }
                uc = sampler.model.get_learned_conditioning(uncondition)

            if 'prompt_embed' not in c['acoustic'].keys():
                cc = c['acoustic']['prompt']
                cc=sampler.model.encode_first_stage(cc)
                zc=sampler.model.get_first_stage_encoding(cc).detach()
                c['acoustic']['prompt_embed']=zc.to(device)

            if 'prompt_embed' not in uc['acoustic'].keys():
                cc = uc['acoustic']['prompt']
                cc=sampler.model.encode_first_stage(cc)
                zc=sampler.model.get_first_stage_encoding(cc).detach()
                uc['acoustic']['prompt_embed']=zc.to(device) 

            samples_ddim, _,f0_pred = sampler.sample_cfg(S=args.ddim_steps,
                                                cond=c,
                                                batch_size=args.n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,
                                                x_T=start_code)
            x_samples_ddim = sampler.model.decode_first_stage(samples_ddim)
            cond_gtcodec_wavs = []
            for idx, spec in enumerate(x_samples_ddim):
                wav = vocoder.vocode(spec.transpose(0, 1).cpu(),f0=f0_pred[idx].squeeze(0)[:spec.shape[1]].cpu())
                cond_gtcodec_wavs.append((spec.cpu(), wav))
            cond_gtcodec_wavs_dict[scale] = cond_gtcodec_wavs
        

        gt_path=item['audio_path']
        gt_wav=sf.read(gt_path)[0]
        
        for scale in scales:
            cond_gtcodec_dir = os.path.join(save_dir, f'cond_gtcodec_scale_{scale}')
            cond_gtcodec_wavs = cond_gtcodec_wavs_dict[scale]
            for wav_idx, (spec, wav) in enumerate(cond_gtcodec_wavs):
                
                cond_gtcodec_path = os.path.join(cond_gtcodec_dir, f"{rank}-{item_idx:04d}[{wav_idx}][pred].wav")
                wav=normalize_loudness(wav, -23)
                sf.write(safe_path(cond_gtcodec_path), wav, 48000, subtype='PCM_16')

                # 保存音频路径和 caption 信息
                csv_data['audio_path'].append(cond_gtcodec_path)  # 音频路径
                csv_data['caption'].append(caption)                       # 对应的 caption
                csv_data['name'].append(item_name)                        # 对应的 item_name
                
                gt_sing_path = os.path.join(cond_gtcodec_dir, f"{rank}-{item_idx:04d}[{wav_idx}][gt].wav")
                gt_sing = normalize_loudness(gt_wav, -23)
                sf.write(safe_path(gt_sing_path), gt_sing,48000, subtype='PCM_16')
                
        results.append(item)
        item_idx += 1
    # 保存CSV文件
    csv_save_path = os.path.join(save_dir, 'clap.csv')
    save_df_to_tsv(pd.DataFrame.from_dict(csv_data),  csv_save_path)

    print(f"CSV 文件保存至: {csv_save_path}")


if __name__ == '__main__':
    args = parse_args()
    args.dist_config = {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54189",
        "world_size": 1
    }
    if args.num_gpus > 1:
        mp.spawn(gen_song, nprocs=args.num_gpus, args=(args,))
    else:
        gen_song(0, args=args)
