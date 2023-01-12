# Deep Natural Language Processing project

Project carried out for the exam of DNLP course at Politecnico di Torino a.y. 2022-2023. Professor: Luca Cagliero, Moreno La Quatra.

## First extension
Thit first extension is a title generation task. The employed model is BART starting from the pretrained version [distilbart](https://huggingface.co/sshleifer/distilbart-cnn-12-6). The input sequence is a concatenation of the highlights and the abstract of the paper via a SEP token.

All the results (Rouge1, Rouge2, BertScore) are available in the results folder. As an example the model fine-tuned on CS gets:
* Rouge1: 0.55
* Rouge2: 0.38
* BertScore: 0.92

on the papers in the CSPubSumm.

## Second Extension
The second extension is a Probabilistic Context Enrichment aiming to improve the performance obtained by [THExt](https://www.sciencedirect.com/science/article/abs/pii/S0950705122006931) ([GitHubRepository](https://github.com/MorenoLaQuatra/THExt)).
