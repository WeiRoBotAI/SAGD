# SAGD

## Spatial Autocorrelation-Aware Geographic Disambiguation for Natural Language-Guided Geo-Localization

Implementation of **SAGD**, a framework for retrieving geo-tagged satellite images or OpenStreetMap (OSM) references from ground-level natural language descriptions.

## Overview

Nearby locations often share road layouts, building arrangements, and urban block patterns. SAGD combines geographically structured candidate ranking with local cross-modal supervision to improve discrimination among similar locations.

- **GeoCMR — Geographic Confusion-Aware Margin Ranking:** constructs geographically eligible negative sets using annular neighborhoods and nearest-neighbor supplementation, expands the available candidates through cross-batch memory, and imposes bidirectional ranking margins.
- **DEG — Discriminative Evidence Grounding:** selects candidate textual cues and compares their local visual responses in the paired image and an independently selected semantic hard negative.
- **FG-CLIP backbone:** provides global retrieval embeddings and dense image/text representations. The long-text branch is extended from 248 to 300 positions while preserving its first 20 positional embeddings.

## Requirements and Installation

Ensure your environment meets the following requirements:

```
conda create -n CVG-Text python=3.9 -y
conda activate CVG-Text
pip install -r requirements.txt
```

## Dataset Download and Path Configuration

**Dataset：** The dataset can be downloaded from [here]([yejy53/CVG-Text: [ICCV 2025\] Where am I? Cross-View Geo-localization with Natural Language Descriptions.](https://github.com/yejy53/CVG-Text)).

**Path Configuration：** After downloading, update the `/path/to/dataset/` in `./config.yaml` with the actual dataset paths.

From the repository root:

**Model Checkpoints：** Our model checkpoints are available at:

## Pretrained Backbone

The model loader uses [qihoo360/fg-clip-large]([360CVGroup/FG-CLIP: New generation of CLIP with strong fine grained discrimination capability, ICML2026 and ICML2025](https://github.com/360CVGroup/FG-CLIP))

To use an offline model, download the complete model repository and update:

```yaml
pretrained_paths:
  fgclip_l14_336: "/path/to/pretrained/fg-clip-large"
```

## Training

The current SAGD training entry point requires **one GPU** and both `--geo_cmr` and `--gfa`. Launch it with `torch.distributed.run`, including for a single GPU.

```
./scripts/train_geoharmony.sh
```

### OSM retrieval and other cities

```bash
CITY=NewYork IMG_TYPE=sat bash scripts/train_geoharmony.sh
CITY=NewYork IMG_TYPE=osm bash scripts/train_geoharmony.sh
CITY=Brisbane IMG_TYPE=sat bash scripts/train_geoharmony.sh
CITY=Brisbane IMG_TYPE=osm bash scripts/train_geoharmony.sh
CITY=Tokyo IMG_TYPE=sat bash scripts/train_geoharmony.sh
CITY=Tokyo IMG_TYPE=osm bash scripts/train_geoharmony.sh
```

To reduce memory use, lower the batch size, for example:

```bash
CITY=NewYork IMG_TYPE=sat BS=10 bash scripts/train_geoharmony.sh
```

## Evaluation

Place a fine-tuned checkpoint under:

```text
checkpoints/FG-CLIP-L_14@336/
```

Then pass its **filename**, without repeating the directory:

```bash
# Replace this with the filename of your actual trained/downloaded checkpoint.
CKPT="long_model_NewYork-mixed_sat_lr1e-05_bs20_GeoHarmony_epoch40_53.67.pth"

python zeroshot.py \
  --version NewYork-mixed \
  --img_type sat \
  --model FG-CLIP-L/14@336 \
  --expand \
  --checkpoint "${CKPT}"
```

The filename above is an example of the existing checkpoint convention, not a bundled checkpoint. Verify that the console reports loading your specified weights. The inherited loader can fall back to pretrained weights when a checkpoint is missing.

Outputs are written under `results/FG-CLIP-L_14@336/`, including top-5 matches in `<checkpoint>.csv` and a plain-text metric summary named `<checkpoint>_metric.csv`.

### Evaluation protocol

The default evaluator selects up to **100 geographically nearest reference entries** using the query's paired coordinates, then ranks that local gallery by global embedding similarity. It reports R@1/R@5/R@10 and L@50/L@100/L@150, where localization thresholds are in meters. This is a local-gallery benchmark protocol, not unrestricted city-wide retrieval. Coordinates affect candidate construction and distance metrics; they do not enter the cosine similarity score.

## Results Reported in the Manuscript

Scores are percentages. This table records the manuscript results; it is not a new evaluation performed by the installation commands above. Use the corresponding checkpoints and logged configurations for exact reproduction.

![image-20261015174232577](.\image-20261015174232577.png)

## Acknowledgments

This implementation builds on [CrossText2Loc / CVG-Text](https://github.com/yejy53/CVG-Text), [FG-CLIP](https://github.com/360CVGroup/FG-CLIP), and [OpenAI CLIP](https://github.com/openai/CLIP). We thank their authors for providing the datasets, pretrained models, and code.

Please cite the corresponding works when using their data or models. Their licenses and release conditions apply to the respective third-party resources.

## Citation

If this work is useful to your research, please cite the manuscript. The entry below contains the current title; the author list and publication metadata should be added when finalized.

```bibtex
@article{ye2024cross,
  title={Where am I? Cross-View Geo-localization with Natural Language Descriptions},
  author={Ye, Junyan and Lin, Honglin and Ou, Leyan and Chen, Dairong and Wang, Zihao and He, Conghui and Li, Weijia},
  journal={arXiv preprint arXiv:2412.17007},
  year={2024}
}
```

```
@inproceedings{xie2025fg2,
  title={FG-CLIP 2: A Bilingual Fine-grained Vision-language Alignment Model},
  author={Xie, Chunyu and Wang, Bin and Kong, Fanjing and Li, Jincheng and Liang, Dawei and Ao, Ji and Leng, Dawei and Yin, Yuhui},
  booktitle={International Conference on Machine Learning},
  year={2026}
}


@inproceedings{xie2025fg,
  title={FG-CLIP: Fine-Grained Visual and Textual Alignment},
  author={Xie, Chunyu and Wang, Bin and Kong, Fanjing and Li, Jincheng and Liang, Dawei and Zhang, Gengshen and Leng, Dawei and Yin, Yuhui},
  booktitle={International Conference on Machine Learning},
  pages={68777--68793},
  year={2025}
}
```
