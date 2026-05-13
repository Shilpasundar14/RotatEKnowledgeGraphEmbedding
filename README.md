# TracIn-CP × RotatE on the ROBOKOP Biomedical Knowledge Graph

> Research fork of [DeepGraphLearning/KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) (the official RotatE implementation from [Sun et al., ICLR 2019](https://openreview.net/forum?id=HkgEQnRqYQ)).
>
> The upstream RotatE README is preserved verbatim [below the horizontal rule](#-upstream-readme-rotate-knowledge-graph-embedding-by-relational-rotation-in-complex-space).

## What this fork adds

This fork extends the upstream RotatE training pipeline with **TracIn-CP** ([Pruthi et al., NeurIPS 2020](https://arxiv.org/abs/2002.08484)) training-data influence analysis, applied to subgraphs of the **ROBOKOP** biomedical knowledge graph ([Bizon et al., RENCI/UNC](https://robokop.renci.org/)). The goal is interpretability: given a *test* triple that RotatE predicts (or fails to predict), identify which *training* triples actually moved the model toward that prediction.

Three subgraphs of increasing schema complexity are studied:

| Subgraph | Node types | Use |
| --- | --- | --- |
| CCD   | Chemical–Chemical–Disease | First TracIn-CP run; baseline |
| CCDD  | Chemical–Chemical–Disease–Disease (adds disease–disease edges) | Co-morbidity structure |
| CCGGDD | Chemical–Chemical–Gene–Gene–Disease–Disease | Full chemical→gene→disease path; the setting used in the headline results |

## TracIn-CP, briefly

For a checkpoint set `{θ_t}` produced during training, the TracIn-CP score of a training triple `z_train` on a test triple `z_test` is

```
TracIn-CP(z_train, z_test) = Σ_t  η_t · ⟨ ∇L(z_train; θ_t),  ∇L(z_test; θ_t) ⟩
```

i.e. the sum, over checkpoints, of dot products between per-example loss gradients. A large positive value means `z_train` *helped* the model predict `z_test`; a large negative value means `z_train` *hurt* that prediction; near-zero values are irrelevant. We compute this with RotatE's `logsigmoid` margin loss and average over the per-parameter gradients of `model.parameters()`.

## Repository layout

```
.
├── codes/                              # upstream RotatE training code
│   ├── run.py                          #   + --cpu_test flag (memory-bound eval)
│   ├── model.py
│   └── dataloader.py
├── compute_tracin_cp.py                # main TracIn-CP score computation
├── analyze_tracin_CP_CCGGDDsubgraph.py # top/bottom influential triples per test
├── benchmark_tracin_one_test.py        # wall-clock benchmark, one test vs all train
├── selfscores.py                       # self-to-self influence sanity check
├── map_tracin_ids_to_names.py          # CHEBI/HP/MONDO IDs → human-readable names
├── plot_rotate_CCDD_results.py         # training/val curves for CCDD
├── plot_rotate_CCGGDD_results.py       # training/val curves for CCGGDD
├── data/
│   ├── rotate_protorobo_CCDD/          # CCDD splits (full)
│   └── rotate_protorobo_CCGGDD/        # CCGGDD splits (train.txt excluded — 327 MB, see below)
├── train_matrix_subgraph_CCDD.slurm    # cluster training scripts
├── train_matrix_subgraph_CCGGDD.slurm
├── run_tracin_cp.slurm                 # TracIn-CP scoring job
├── rotate_CCDD.log, rotate_CCGGDD.log  # full training logs
├── rotate_CCDD_*.png, rotate_CCGGDD_*.png   # MRR / MR / HITS@10 / loss curves
└── tracin_cp_scores_*.csv              # raw influence scores
```

## Reproducing the headline (CCGGDD) experiment

> The CCGGDD `train.txt` is 327 MB and exceeds GitHub's 100 MB file limit. Regenerate it from ROBOKOP using `convert_rotorobo_to_rotate.py` (the script consumes the raw `rotorobo.txt` export — pointers in the script docstring). The entity/relation dictionaries and the `valid.txt` / `test.txt` splits are committed.

1. **Train RotatE on CCGGDD, saving multiple checkpoints** (this is what TracIn-CP needs — it sums dot products across `t`):
   ```
   sbatch train_matrix_subgraph_CCGGDD.slurm
   ```
   Output: `models/RotatE_CCGGDD/checkpoint_*.pt` and `rotate_CCGGDD_*.png` plots.

2. **Compute TracIn-CP influence scores** for selected test triples against the training set:
   ```
   sbatch run_tracin_cp.slurm   # wraps compute_tracin_cp.py
   ```
   Output: `tracin_cp_scores_CCGGDD_models2_10_26.csv` (one row per `(checkpoint, train_triple, test_triple)`).

3. **Map opaque biomedical IDs to readable names** (CHEBI:5781 → "chlorothiazide", HP:0002239 → "gingival bleeding", etc.):
   ```
   python map_tracin_ids_to_names.py
   ```
   Output: `tracin_cp_scores_CCGGDD_models2_10_26_mapped.csv`.

4. **Analyze** — find the test triples with the most diverse influential training relations, and the most influential / counter-influential training triples per test:
   ```
   python analyze_tracin_CP_CCGGDDsubgraph.py
   ```

## Computational notes

- TracIn-CP cost is `O(|test| · |train| · |checkpoints|)` gradient evaluations. `benchmark_tracin_one_test.py` measures one test triple against the full CCGGDD train set and reports the projected cost for 10 000 test triples — useful before launching a multi-day job.
- For the CCGGDD model we used checkpoint indices `{2, 10, 26}` (early / mid / late training), which is the configuration baked into `tracin_cp_scores_CCGGDD_models2_10_26.csv`.
- `--cpu_test` in `codes/run.py` evaluates on CPU after each training block; useful when the CCGGDD entity embedding table doesn't fit alongside a full evaluation batch on a single GPU.

## Citations

If you use this fork, please cite both the upstream RotatE paper and the TracIn-CP paper:

```bibtex
@inproceedings{sun2019rotate,
  title     = {RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space},
  author    = {Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019},
  url       = {https://openreview.net/forum?id=HkgEQnRqYQ}
}

@inproceedings{pruthi2020tracin,
  title     = {Estimating Training Data Influence by Tracing Gradient Descent},
  author    = {Garima Pruthi and Frederick Liu and Satyen Kale and Mukund Sundararajan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020},
  url       = {https://arxiv.org/abs/2002.08484}
}
```

---

# 📄 Upstream README: RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space

**Introduction**

This is the PyTorch implementation of the [RotatE](https://openreview.net/forum?id=HkgEQnRqYQ) model for knowledge graph embedding (KGE). We provide a toolkit that gives state-of-the-art performance of several popular KGE models. The toolkit is quite efficient, which is able to train a large KGE model within a few hours on a single GPU.

A faster multi-GPU implementation of RotatE and other KGE models is available in [GraphVite](https://github.com/DeepGraphLearning/graphvite).

**Implemented features**

Models:
 - [x] RotatE
 - [x] pRotatE
 - [x] TransE
 - [x] ComplEx
 - [x] DistMult

Evaluation Metrics:

 - [x] MRR, MR, HITS@1, HITS@3, HITS@10 (filtered)
 - [x] AUC-PR (for Countries data sets)

Loss Function:

 - [x] Uniform Negative Sampling
 - [x] Self-Adversarial Negative Sampling

**Usage**

Knowledge Graph Data:
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE model is evaluated on this data set

**Train**

For example, this command train a RotatE model on FB15k dataset with GPU 0.
```
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data/FB15k \
 --model RotatE \
 -n 256 -b 1024 -d 1000 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.0001 --max_steps 150000 \
 -save models/RotatE_FB15k_0 --test_batch_size 16 -de
```
   Check argparse configuration at codes/run.py for more arguments and more details.

**Test**

    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE

**Reproducing the best results**

To reprocude the results in the ICLR 2019 paper [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://openreview.net/forum?id=HkgEQnRqYQ), you can run the bash commands in best_config.sh to get the best performance of RotatE, TransE, and ComplEx on five widely used datasets (FB15k, FB15k-237, wn18, wn18rr, Countries).

The run.sh script provides an easy way to search hyper-parameters:

    bash run.sh train RotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 200000 16 -de

**Speed**

The KGE models usually take about half an hour to run 10000 steps on a single GeForce GTX 1080 Ti GPU with default configuration. And these models need different max_steps to converge on different data sets:

| Dataset | FB15k | FB15k-237 | wn18 | wn18rr | Countries S* |
|-------------|-------------|-------------|-------------|-------------|-------------|
|MAX_STEPS| 150000 | 100000 | 80000 | 80000 | 40000 | 
|TIME| 9 h | 6 h | 4 h | 4 h | 2 h | 

**Results of the RotatE model**

| Dataset | FB15k | FB15k-237 | wn18 | wn18rr |
|-------------|-------------|-------------|-------------|-------------|
| MRR | .797 ± .001 | .337 ± .001 | .949 ± .000 |.477 ± .001
| MR | 40 | 177 | 309 | 3340 |
| HITS@1 | .746 | .241 | .944 | .428 |
| HITS@3 | .830 | .375 | .952 | .492 |
| HITS@10 | .884 | .533 | .959 | .571 |

**Using the library**

The python libarary is organized around 3 objects:

 - TrainDataset (dataloader.py): prepare data stream for training
 - TestDataSet (dataloader.py): prepare data stream for evluation
 - KGEModel (model.py): calculate triple score and provide train/test API

The run.py file contains the main function, which parses arguments, reads data, initilize the model and provides the training loop.

Add your own model to model.py like:
```
def TransE(self, head, relation, tail, mode):
    if mode == 'head-batch':
        score = head + (relation - tail)
    else:
        score = (head + relation) - tail

    score = self.gamma.item() - torch.norm(score, p=1, dim=2)
    return score
```

**Citation**

If you use the codes, please cite the following [paper](https://openreview.net/forum?id=HkgEQnRqYQ):

```
@inproceedings{
 sun2018rotate,
 title={RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space},
 author={Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang},
 booktitle={International Conference on Learning Representations},
 year={2019},
 url={https://openreview.net/forum?id=HkgEQnRqYQ},
}
```
