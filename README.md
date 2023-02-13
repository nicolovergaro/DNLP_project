# Deep Natural Language Processing project

Project carried out for the exam of DNLP course at Politecnico di Torino a.y. 2022-2023.

Professor: Luca Cagliero, Moreno La Quatra.

## First extension
Thit first extension is a title generation task. The employed model is BART starting from the pretrained version [distilbart](https://huggingface.co/sshleifer/distilbart-cnn-12-6). The input sequence is a concatenation of the highlights and the abstract of the paper via a SEP token.

|                                     | Rouge-1 F1 | Rouge-2 F1 | Rouge-L F1 | BERTScore F1 |
|:-----------------------------------:|:----------:|:----------:|:----------:|:------------:|
|  bart-paper-titels-ai on AIPubSumm  |   0.4332   |   0.2240   |   0.3607   |    0.9064    |
| bart-paper-titles-bio on BIOPubSumm |   0.4580   |   0.2541   |   0.3961   |    0.9027    |
|   bart-paper-title-cs on CSPubSumm  |   0.5584   |   0.3818   |   0.5012   |    0.9233    |

You can find a demo hosted on [HF spaces](https://huggingface.co/spaces/pietrocagnasso/paper-title-generation) to see all the models in action.

## Second Extension
The second extension is a Probabilistic Context Extraction aiming to improve the performance obtained by [THExt](https://www.sciencedirect.com/science/article/abs/pii/S0950705122006931) ([GitHub repo](https://github.com/MorenoLaQuatra/THExt)).

## Requirements
Our experiments were carried out using the following packages:
* transformers 4.25.1
* torch 1.11.0
* py-rouge 1.1
* evaluate 0.4.0
* bert_score 0.3.12
