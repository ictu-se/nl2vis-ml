# Model Comparison Template

## Main Table
| Model | Params | Trainable Params | Src Vocab | Trg Vocab | Best Dev Loss | Test EM | Slot F1 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Train Time (s) | Inference ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RNN |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LSTM |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| GRU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Transformer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Notes
- Use identical data split and seed policy for fairness.
- Report mean/std across seeds if available.
- Keep decoding strategy fixed (greedy or beam) across models.
