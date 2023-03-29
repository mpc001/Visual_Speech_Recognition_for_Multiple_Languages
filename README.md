<p align="center"><img width="160" src="doc/lip_white.png" alt="logo"></p>
<h1 align="center">Visual Speech Recognition for Multiple Languages</h1>

<div align="center">

[📘Introduction](#Introduction) |
[🛠️Preparation](#Preparation) |
[📊Benchmark](#Benchmark-evaluation) |
[🔮Inference](#Speech-prediction) |
[🐯Model zoo](#Model-Zoo) |
[📝License](#License)
</div>

## Authors

[Pingchuan Ma](https://mpc001.github.io/), [Alexandros Haliassos](https://dblp.org/pid/257/3052.html), [Adriana Fernandez-Lopez](https://scholar.google.com/citations?user=DiVeQHkAAAAJ), [Honglie Chen](https://scholar.google.com/citations?user=HPwdvwEAAAAJ), [Stavros Petridis](https://ibug.doc.ic.ac.uk/people/spetridis), [Maja Pantic](https://ibug.doc.ic.ac.uk/people/mpantic).

## Update

`2023-03-27`: We have released our AutoAVSR models for LRS3, see [here](#autoavsr-models).

## Introduction

This is the repository of [Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels](https://arxiv.org/abs/2303.14307) and [Visual Speech Recognition for Multiple Languages](https://arxiv.org/abs/2202.13084), which is the successor of [End-to-End Audio-Visual Speech Recognition with Conformers](https://arxiv.org/abs/2102.06657). By using this repository, you can achieve the performance of 19.1%, 1.0% and 0.9% WER for automatic, visual, and audio-visual speech recognition (ASR, VSR, and AV-ASR) on LRS3.

## Demo

English -> Mandarin -> Spanish   |    French -> Portuguese -> Italian  |
:-------------------------------:|:------------------------------------:
<img src='doc/vsr_1.gif' title='vsr1' style='max-width:320px'></img>  |  <img src='doc/vsr_2.gif' title='vsr2' style='max-width:320px'></img>  |

<div align="center">

[Youtube](https://youtu.be/FIau-6JA9Po) |
[Bilibili](https://www.bilibili.com/video/BV1Wu411D7oP)
</div>


## Preparation
1. Clone the repository and enter it locally:

```Shell
git clone https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
cd Visual_Speech_Recognition_for_Multiple_Languages
```

2. Setup the environment.
```Shell
conda create -y -n autoavsr python=3.8
conda activate autoavsr
```

3. Install pytorch, torchvision, and torchaudio by following instructions [here](https://pytorch.org/get-started/), and install all packages:

```Shell
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```

4. Download and extract a pre-trained model and/or language model from [model zoo](#Model-Zoo) to:

- `./benchmarks/${dataset}/models`

- `./benchmarks/${dataset}/language_models`

5. [For VSR and AV-ASR] Install [RetinaFace](./tools) or [MediaPipe](https://pypi.org/project/mediapipe/) tracker.

### Benchmark evaluation

```Shell
python eval.py config_filename=[config_filename] \
               labels_filename=[labels_filename] \
               data_dir=[data_dir] \
               landmarks_dir=[landmarks_dir]
```

- `[config_filename]` is the model configuration path, located in `./configs`.

- `[labels_filename]` is the labels path, located in `${lipreading_root}/benchmarks/${dataset}/labels`.

- `[data_dir]` and `[landmarks_dir]` are the directories for original dataset and corresponding landmarks.

- `gpu_idx=-1` can be added to switch from `cuda:0` to `cpu`.

### Speech prediction

```Shell
python infer.py config_filename=[config_filename] data_filename=[data_filename]
```

- `data_filename` is the path to the audio/video file.

- `detector=mediapipe` can be added to switch from RetinaFace to MediaPipe tracker.

### Mouth ROIs cropping

```Shell
python main.py data_filename=[data_filename] dst_filename=[dst_filename]
```

- `dst_filename` is the path where the cropped mouth will be saved.

## Model zoo

### Overview

We support a number of datasets for speech recognition:
- [x] [Lip Reading Sentences 2 (LRS2)](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)
- [x] [Lip Reading Sentences 3 (LRS3)](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html)
- [x] [Chinese Mandarin Lip Reading (CMLR)](https://www.vipazoo.cn/CMLR.html)
- [x] [CMU Multimodal Opinion Sentiment, Emotions and Attributes (CMU-MOSEAS)](http://immortal.multicomp.cs.cmu.edu/cache/multilingual)
- [x] [GRID](http://spandh.dcs.shef.ac.uk/gridcorpus)
- [x] [Lombard GRID](http://spandh.dcs.shef.ac.uk/avlombard)
- [x] [TCD-TIMIT](https://sigmedia.tcd.ie)

### AutoAVSR models

<details open>

<summary>Lip Reading Sentences 3 (LRS3)</summary>

<p> </p>

|     Components        |  WER |                                             url                                         |  size (MB)  |
|:----------------------|:----:|:---------------------------------------------------------------------------------------:|:-----------:|
|   **Visual-only**     |
| -                     | 19.1 |[GoogleDrive](http://bit.ly/40EAtyX) or [BaiduDrive](https://bit.ly/3ZjbrV5)(key: dqsy)  |     891     |
|   **Audio-only**      |
| -                     | 1.0  |[GoogleDrive](http://bit.ly/3ZSdh0l) or [BaiduDrive](http://bit.ly/3Z1TlGU)(key: dvf2)   |     860     |
|   **Audio-visual**    |
| -                     | 0.9  |[GoogleDrive](http://bit.ly/3yRSXAn) or [BaiduDrive](http://bit.ly/3LAxcMY)(key: sai5)   |     1540    |
| **Language models**   |
| -                     |   -  |[GoogleDrive](http://bit.ly/3FE4XsV) or [BaiduDrive](http://bit.ly/3yRI5SY)(key: t9ep)   |     191     |
| **Landmarks**         |
| -                     |   -  |[GoogleDrive](https://bit.ly/33rEsax) or [BaiduDrive](https://bit.ly/3rwQSph)(key: mi3c) |     18577   |

</details>

### VSR for multiple languages models

<details open>

<summary>Lip Reading Sentences 2 (LRS2)</summary>

<p> </p>

|     Components        |  WER |                                             url                                         |  size (MB)  |
|:----------------------|:----:|:---------------------------------------------------------------------------------------:|:-----------:|
|   **Visual-only**     |
| -                     | 26.1 |[GoogleDrive](https://bit.ly/3I25zrH) or [BaiduDrive](https://bit.ly/3BAHBkH)(key: 48l1) |     186     |
| **Language models**   |
| -                     |   -  |[GoogleDrive](https://bit.ly/3qzWKit) or [BaiduDrive](https://bit.ly/3KgAL7T)(key: 59u2) |     180     |
| **Landmarks**         |
| -                     |   -  |[GoogleDrive](https://bit.ly/3jSMMoz) or [BaiduDrive](https://bit.ly/3BuIwBB)(key: 53rc) |     9358    |

</details>


<details open>

<summary>Lip Reading Sentences 3 (LRS3)</summary>

<p> </p>

|     Components        |  WER |                                             url                                         |  size (MB)  |
|:----------------------|:----:|:---------------------------------------------------------------------------------------:|:-----------:|
|   **Visual-only**     |
| -                     | 32.3 |[GoogleDrive](https://bit.ly/3Bp4gjV) or [BaiduDrive](https://bit.ly/3rIzLCn)(key: 1b1s) |     186     |
| **Language models**   |
| -                     |   -  |[GoogleDrive](https://bit.ly/3qzWKit) or [BaiduDrive](https://bit.ly/3KgAL7T)(key: 59u2) |     180     |
| **Landmarks**         |
| -                     |   -  |[GoogleDrive](https://bit.ly/33rEsax) or [BaiduDrive](https://bit.ly/3rwQSph)(key: mi3c) |     18577   |

</details>



<details open>

<summary>Chinese Mandarin Lip Reading (CMLR)</summary>

<p> </p>

|     Components        |  CER |                                             url                                         |  size (MB)  |
|:----------------------|:----:|:---------------------------------------------------------------------------------------:|:-----------:|
|   **Visual-only**     |
| -                     |  8.0 |[GoogleDrive](https://bit.ly/3fR8RkU) or [BaiduDrive](https://bit.ly/3IyACLB)(key: 7eq1) |     195     |
| **Language models**   |
| -                     |   -  |[GoogleDrive](https://bit.ly/3fPxXAJ) or [BaiduDrive](https://bit.ly/3rEcErr)(key: k8iv) |     187     |
| **Landmarks**         |
| -                     |   -  |[GoogleDrive](https://bit.ly/3bvetPL) or [BaiduDrive](https://bit.ly/3o2u53d)(key: 1ret) |     3721    |

</details>


<details open>

<summary>CMU Multimodal Opinion Sentiment, Emotions and Attributes (CMU-MOSEAS)</summary>

<p> </p>

|     Components        |  WER |                                             url                                         |  size (MB)  |
|:----------------------|:----:|:---------------------------------------------------------------------------------------:|:-----------:|
|   **Visual-only**     |
| Spanish               | 44.5 |[GoogleDrive](https://bit.ly/34MjWBW) or [BaiduDrive](https://bit.ly/33rMq3a)(key: m35h) |     186     |
| Portuguese            | 51.4 |[GoogleDrive](https://bit.ly/3HjXCgo) or [BaiduDrive](https://bit.ly/3IqbbMg)(key: wk2h) |     186     |
| French                | 58.6 |[GoogleDrive](https://bit.ly/3Ik6owb) or [BaiduDrive](https://bit.ly/35msiQG)(key: t1hf) |     186     |
| **Language models**   |
| Spanish               |   -  |[GoogleDrive](https://bit.ly/3rppyJN) or [BaiduDrive](https://bit.ly/3nA3wCN)(key: 0mii) |     180     |
| Portuguese            |   -  |[GoogleDrive](https://bit.ly/3gPvneF) or [BaiduDrive](https://bit.ly/33vL8Es)(key: l6ag) |     179     |
| French                |   -  |[GoogleDrive](https://bit.ly/3LDChSn) or [BaiduDrive](https://bit.ly/3sNnNql)(key: 6tan) |     179     |
| **Landmarks**         |
| -                     |   -  |[GoogleDrive](https://bit.ly/34Cf6ak) or [BaiduDrive](https://bit.ly/3BiFG4c)(key: vsic) |     3040    |


</details>


<details open>

<summary>GRID</summary>

<p> </p>

|     Components        |  WER |                                             url                                         |  size (MB)  |
|:----------------------|:----:|:---------------------------------------------------------------------------------------:|:-----------:|
|   **Visual-only**     |
| Overlapped            |  1.2 |[GoogleDrive](https://bit.ly/3Aa6PWn) or [BaiduDrive](https://bit.ly/3IdamGh)(key: d8d2) |     186     |
| Unseen                |  4.8 |[GoogleDrive](https://bit.ly/3patMVh) or [BaiduDrive](https://bit.ly/3t6459A)(key: ttsh) |     186     |
| **Landmarks**         |
| -                     |   -  |[GoogleDrive](https://bit.ly/2Yzu1PF) or [BaiduDrive](https://bit.ly/30fucjG)(key: 16l9) |     1141    |

You can include `data_ext=.mpg` in your command line to match the video file extension in the GRID dataset.

</details>


<details open>

<summary>Lombard GRID</summary>

<p> </p>

|     Components        |  WER |                                             url                                         |  size (MB)  |
|:----------------------|:----:|:---------------------------------------------------------------------------------------:|:-----------:|
|   **Visual-only**     |
| Unseen (Front Plain)  |  4.9 |[GoogleDrive](https://bit.ly/3H5zkGQ) or [BaiduDrive](https://bit.ly/3LE1xI6)(key: 38ds) |     186     |
| Unseen (Side Plain)   |  8.0 |[GoogleDrive](https://bit.ly/3BsGOSO) or [BaiduDrive](https://bit.ly/3sRZYNY)(key: k6m0) |     186     |
| **Landmarks**         |
| -                     |   -  |[GoogleDrive](https://bit.ly/354YOH0) or [BaiduDrive](https://bit.ly/3oWUCA4)(key: cusv) |     309     |

You can include `data_ext=.mov` in your command line to match the video file extension in the Lombard GRID dataset.

</details>


<details open>

<summary>TCD-TIMIT</summary>

<p> </p>

|     Components        |  WER |                                             url                                         |  size (MB)  |
|:----------------------|:----:|:---------------------------------------------------------------------------------------:|:-----------:|
|   **Visual-only**     |
| Overlapped            | 16.9 |[GoogleDrive](https://bit.ly/3Fv7u61) or [BaiduDrive](https://bit.ly/33rPlZN)(key: jh65) |     186     |
| Unseen                | 21.8 |[GoogleDrive](https://bit.ly/3530d0N) or [BaiduDrive](https://bit.ly/3nxZjzC)(key: n2gr) |     186     |
| **Language models**   |
| -                     |   -  |[GoogleDrive](https://bit.ly/3qzWKit) or [BaiduDrive](https://bit.ly/3KgAL7T)(key: 59u2) |     180     |
| **Landmarks**         |
| -                     |   -  |[GoogleDrive](https://bit.ly/3HYmifr) or [BaiduDrive](https://bit.ly/3JFJ6RH)(key: bnm8) |     930     |

</details>


## Citation

If you use the AutoAVSR models, please consider citing the following paper:

```bibtex
@inproceedings{ma2023auto,
  author={Ma, Pingchuan and Haliassos, Alexandros and Fernandez-Lopez, Adriana and Chen, Honglie and Petridis, Stavros and Pantic, Maja},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels}, 
  year={2023},
}
```

If you use the VSR models for multiple languages please consider citing the following paper:

```bibtex
@article{ma2022visual,
  title={{Visual Speech Recognition for Multiple Languages in the Wild}},
  author={Ma, Pingchuan and Petridis, Stavros and Pantic, Maja},
  journal={{Nature Machine Intelligence}},
  volume={4},
  pages={930--939},
  year={2022}
  url={https://doi.org/10.1038/s42256-022-00550-z},
  doi={10.1038/s42256-022-00550-z}
}
```

## License

It is noted that the code can only be used for comparative or benchmarking purposes. Users can only use code supplied under a [License](./LICENSE) for non-commercial purposes.

## Contact

```
[Pingchuan Ma](pingchuan.ma16[at]imperial.ac.uk)
```
