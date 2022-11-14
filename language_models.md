# List of Language models with precision information

The original list of language models with pretraining precision information was retrieved from Stas Bekman's post (https://discuss.huggingface.co/t/model-pre-training-precision-database-fp16-fp32-bf16/5671)

- float16 (mixed precision)
  - allenai/longformer - paper 4, “we employed mixed precision training (floating points 16 and 32) using apex12 to reduce memory consumption and speed-up training. However, we kept the attention computation in fp32 to avoid numerical instability issues.”
  - allenai/led - same as allenai/longformer
  - lvwerra/codeparrot - informed by the creator of the model 1
  - facebook/m2m100_418M (and others) train info 3
  - eleutherai/gpt-neox-20b (doesn’t exist yet, but including for the sake of future-proofing) - as shown in the configs 4. The paper also states that the model was in fp16, see “Appendix B: Full Configuration Details.” Finally, Stella Biderman’s official announcement on Twitter 1 also includes a link to download both “full” weights and “slim” weights which implies mixed precision was used
- bfloat16
  - google/mobilebert - paper 5, “we train IB-BERTLARGE on 256 TPU v3 chips”
  - eleutherai/gpt-neo-1.3b - shown in the config file 11
  - eleutherai/gpt-j-6b - shown in the GitHub readme 13
  - google/pegasus-cnn_dailymail - XXX: needs reference
  - google/pegasus-xsum - XXX: needs reference
  - google/mt5 - most likely same as t5
  - t5 - paper 12 “TPU v3 chips”
  - scifive  paper arXiv:2106.03598 p.2 "TPU v2-8 on Google Colab"
  - bigscience/T0 and other T0* models (trained on TPUs, confirmed on bigscience slack)

- float32
  - float32 (full precision)
  - EleutherAI/gpt-neo-2.7B - the model’s config file 4 doesn’t specify precision and the codebase 2 defaults to fp32
  - gsarti/it5-base 1 and other it5-* - stated by creator 4 (JAX-trained)

## References
Stas Bekman, Model pre-training precision database: fp16, fp32, bf16
https://discuss.huggingface.co/t/model-pre-training-precision-database-fp16-fp32-bf16/5671
