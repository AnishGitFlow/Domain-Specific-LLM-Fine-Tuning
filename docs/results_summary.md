# Results Summary

## Experiment

- Domain: Biomedical research question answering
- Dataset: PubMedQA instruction-style data
- Base model: Qwen/Qwen2.5-1.5B-Instruct
- Fine-tuning method: QLoRA
- Evaluation set: 100 held-out test samples

## Benchmark Table

| Metric | Base Model | Fine-Tuned Model | Change |
|---|---:|---:|---:|
| BLEU | 3.266 | 4.837 | +1.571 |
| ROUGE-1 | 0.273 | 0.302 | +0.029 |
| ROUGE-2 | 0.072 | 0.108 | +0.036 |
| ROUGE-L | 0.167 | 0.216 | +0.049 |
| Perplexity | 10.342 | 9.340 | -1.003 |
| Mean latency | 4.206s | 2.155s | -2.052s |
| p50 latency | 4.146s | 2.120s | -2.026s |
| p95 latency | 7.147s | 3.250s | -3.897s |
| Peak memory max | 1236.874 MB | 1307.312 MB | +70.438 MB |

## Interpretation

The fine-tuned model improves BLEU, ROUGE-1, ROUGE-2, and ROUGE-L, showing that QLoRA adaptation makes the model's responses more aligned with biomedical reference answers. Perplexity also decreases, suggesting better fit to the domain-specific answer distribution.

Latency improves in the benchmark run, likely because the fine-tuned model produces more direct answers. GPU memory usage increases slightly because the LoRA adapter is loaded in addition to the quantized base model.

## Report-Ready Conclusion

QLoRA fine-tuning improved domain-specific biomedical QA performance while keeping training and inference feasible on consumer-grade hardware. The result supports the project hypothesis that parameter-efficient fine-tuning can adapt open-source LLMs to specialized domains without requiring full model fine-tuning or paid APIs.
