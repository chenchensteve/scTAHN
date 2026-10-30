# scTAHN

**scTAHN** (single-cell topology-aware hard negative contrastive clustering) combines topology-aware negative filtering with PPO-adaptive hard-negative sampling for single-cell RNA-seq clustering.

## Environment

| Component | Version |
| --- | --- |
| Python | 3.9 |
| PyTorch | 2.0.0 |
| CUDA | 11.7 |
| PyTorch Geometric | 2.6.1 |
| Scanpy | 1.10.3 |

Install the dependencies listed in [requirements.txt](requirements.txt):

```bash
python -m pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
python -m pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

## Datasets

Place the h5ad files in `./data/`.

| Dataset | Filename | Annotation column | Public source |
| --- | --- | --- | --- |
| Biase | `biase.h5ad` | `cluster` | [GSE57249](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57249), Biase et al. (2014) |
| Klein | `Klein.h5ad` | `cluster` | [GSE65525](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525), Klein et al. (2015) |
| Camp | `Camp.h5ad` | `cluster` | [GSE81252](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81252), Camp et al. (2017) |
| Deng | `deng.h5ad` | `cluster` | [Deng et al. (2014)](https://doi.org/10.1126/science.1245316) |
| Pollen | `Pollen.h5ad` | `cluster` | [Pollen et al. (2014), associated SRA study SRP041736](https://pubmed.ncbi.nlm.nih.gov/25086649/) |
| CEL-seq2-5cl | `CelSeq2_5cl_processed.h5ad` | `cell` | [Tian et al. mixture-control data and scripts](https://github.com/LuyiTian/sc_mixology) |
| 10x-5cl | `10x_5cl_processed.h5ad` | `cell` | [Tian et al. mixture-control data and scripts](https://github.com/LuyiTian/sc_mixology) |
| 10x-LM | `Quake_10x_Limb_Muscle.h5ad` | `cluster` | [Tabula Muris, GSE109774](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774), limb muscle, droplet |
| SS2-LM | `Quake_Smart-seq2_Limb_Muscle.h5ad` | `cluster` | [Tabula Muris, GSE109774](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774), limb muscle, Smart-seq2 |
| SS2-Lung | `Quake_Smart-seq2_Lung.h5ad` | `celltype` | [Tabula Muris, GSE109774](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774), lung, Smart-seq2 |
| SS2-Heart | `Quake_Smart-seq2_Heart.h5ad` | `cluster` | [Tabula Muris, GSE109774](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774), heart, Smart-seq2 |
| 10x-Bladder | `Quake_10x_Bladder.h5ad` | `cluster` | [Tabula Muris, GSE109774](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774), bladder, droplet |
