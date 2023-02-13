# Deep Natural Language Processing project

Project carried out for the exam of DNLP course at Politecnico di Torino a.y. 2022-2023.

Professor: Luca Cagliero, Moreno La Quatra.

## First extension
Thit first extension is a title generation task. The employed model is BART starting from the pretrained version [distilbart](https://huggingface.co/sshleifer/distilbart-cnn-12-6). We have fine-tuned this model for one epoch on all the three datasets available to let the model learn how the title of scientific papers are built, then a second fine-tuning epoch was performed on each dataset to specialize the models.

The following are the results obtained:
|                       | Rouge-1 F1 | Rouge-2 F1 | Rouge-L F1 | BERTScore F1 |
|:---------------------:|:----------:|:----------:|:----------:|:------------:|
|  bpt-ai on AIPubSumm  |   0.4332   |   0.2240   |   0.3607   |    0.9064    |
| bpt-bio on BIOPubSumm |   0.4580   |   0.2541   |   0.3961   |    0.9027    |
|  bpt-cs on CSPubSumm  |   0.5584   |   0.3818   |   0.5012   |    0.9233    |

You can find our [ai](https://huggingface.co/pietrocagnasso/bart-paper-titles-ai), [bio](https://huggingface.co/pietrocagnasso/bart-paper-titles-bio), [cs](https://huggingface.co/pietrocagnasso/bart-paper-titles-cs) models on HuggingFace and see them in action in the demo we set up on [HF spaces](https://huggingface.co/spaces/pietrocagnasso/paper-title-generation).

## Second Extension
The second extension is a Probabilistic Context Extraction aiming to improve the performance obtained by [THExt](https://www.sciencedirect.com/science/article/abs/pii/S0950705122006931) ([GitHub repo](https://github.com/MorenoLaQuatra/THExt)) changing the context. Starting from the models provided with the paper we built our custom context using all the 3 picking strategies we defined and trained the models for an additional epoch using thiese contexts.

The following are the results obtained using the picking strategy called "best":
|                        | Rouge-1 F1 | Rouge-2 F1 | Rouge-L F1 |
|:----------------------:|:----------:|:----------:|:----------:|
|  PCE-best on AIPubSumm |   0.3415   |   0.1250   |   0.3111   |
| PCE-best on BIOPubSumm |   0.3335   |   0.1222   |   0.3038   |
|  PCE-best on CSPubSumm |   0.3738   |   0.1613   |   0.3443   |

You can find our [ai](https://huggingface.co/pietrocagnasso/thext-pce-ai), [bio](https://huggingface.co/pietrocagnasso/thext-pce-bio), [cs](https://huggingface.co/pietrocagnasso/thext-pce-cs) models on HuggingFace and see them in action in the demo we set up on [HF spaces](https://huggingface.co/spaces/pietrocagnasso/paper-highlights-extraction).
