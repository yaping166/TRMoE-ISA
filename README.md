# TRMoE-ISA

Implementation for our preprinted paper **Task-Routed Mixture-of-Experts with Cognitive
Appraisal for Implicit Sentiment Analysis** ([paper](https://arxiv.org/pdf/2605.20916v1)).

## Project Structure

```
main.py                     argument parsing
run.py                      data + model + trainer
modules/
  ffn_moe.py                task-routed FFN-MoE
  trainer.py                Seq2SeqTrainer subclass, collator, best-model callback
isa_data/
  isa.py                    dataset conversion, prompts, tokenization
  metrics.py                polarity / implicit / rationale metrics
utils.py                    seed, layer-list parsing, decoding, disk cleanup
```



## Data

We follow the implicit sentiment dataset splits of [SCAPT-ABSA](https://github.com/Tribleave/SCAPT-ABSA).


## Environment

Developed with Python 3.11.

- torch 2.7.0
- transformers 4.57.3
- datasets 4.4.1
- numpy 1.26.4
- scikit-learn 1.4.1

Install the dependencies with:

```bash
pip install -r requirements.txt
```


## Training and Evaluation

Using the Laptop14 dataset as a running example,


```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --train_path=MTL_datasets/Lap/Lap_Train.json \
  --test_path=MTL_datasets/Lap/Lap_Test.json \
  --output_dir=results/Lap \
  --do_train \
  --do_predict
```

## Important Arguments

- `--num_experts`: number of experts per MoE layer.
- `--top_k`: top-k experts selected per sample.
- `--encoder_moe_layers`, `--decoder_moe_layers`: block indices that adopt MoE FFNs.
- `--task_embedding_dim`, `--router_hidden_dim`: task router size.
- `--lambda_sep`: weight of the task-separated routing objective.

## Citation

If you find our code and paper useful, please kindly cite us using the BibTeX below.

```bibtex
@misc{chai2026taskrouted,
      title={Task-Routed Mixture-of-Experts with Cognitive Appraisal for Implicit Sentiment Analysis}, 
      author={Yaping Chai and Haoran Xie and Joe S. Qin},
      year={2026},
      howpublished = {arXiv:2605.20916},
}

```

## Acknowledgments

This code is partly referred from [CrossTaskMoE](https://github.com/INK-USC/CrossTaskMoE). We thank the authors of [CrossTaskMoE](https://github.com/INK-USC/CrossTaskMoE) for their open-source release.
