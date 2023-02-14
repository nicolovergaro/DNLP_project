# Code for extension 2: Probabilistic Context Extraction for THExt

import torch
import json
import numpy as np

from tqdm import tqdm
from rouge import Rouge
from itertools import chain
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

from utils.reproducibility import *
from utils.datasets import PCEDataset


class InternalDataset(Dataset):
    def __init__(self, sentences, context, tokenizer):
        self.x = sentences
        self.context = context
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        x = self.tokenizer.encode_plus(self.x[index],
                                       self.context,
                                       padding="max_length",
                                       truncation=True,
                                       max_length=384,
                                       return_attention_mask=True,
                                       return_tensors='pt')
        
        return {"input_ids": x["input_ids"][0],
                "attention_mask": x["attention_mask"][0]}
        
    def __len__(self):
        return len(self.x)


class Highlighter():
    def __init__(self, model_name="pietrocagnasso/thext-pce-cs"):
        """
        The models we fine-tuned for this extensions are available on huggingface:
        - pietrocagnasso/thext-pce-ai: strating from morenolq/thext-ai-scibert fine-tuned for
                another epoch on AIPubSumm
        - pietrocagnasso/thext-pce-bio: strating from morenolq/thext-bio-scibert fine-tuned for
                another epoch on BIOPubSumm
        - pietrocagnasso/thext-pce-cs: strating from morenolq/thext-cs-scibert fine-tuned for
                another epoch on CSPubSumm
                
        This class is fully compatible with the models proposed by La Quatra, Cagliero "Transformer-
        based highlights extraction from scientific papers" from which we started our fine-tuning.

        Parameters:
            model_name: string with the name of the model to be used, by default it is the model
                trained on all the datasets
        """

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    def train(self, 
              file,
              contributions_file,
              distributions_file,
              topic,
              context_length=15,
              policy="random",
              sep_by_sent=True,
              epochs=1,
              per_device_batch_size=16,
              n_gpus=1,
              lr=1e-5,
              weight_decay=1e-2,
              model_output_dir="./best_model",
              seed=None
              ):
        """
        This method can be used to tune a model starting from scratch or fine-tuning a pre-trained
        model. The dataset passed to the function can be either a pytrch dataset (PCEDataset) or a
        JSON file following the requirements given by PCEDataset.

        Parameters:
            file: path to a file JSON or pt with a dataset
            epochs: number of training epochs
            per_device_batch_size: size f the batch for each device
            n_gpus: number of gpus avilable in the system
            lr: learning rate
            weight_decay: weight_decay
            model_output_dir: directory in which the best model should be saved
            seed: seed for reproducibility purposes (take a look to utils.reproducibility)
        """

        # reproducibility
        if seed is not None:
            make_it_reproducible(seed)

        # build the datasets
        print("--- BUILDING THE DATASET ---")
        if file.split(".")[-1] == "json":
            train_ds = PCEDataset(file=file,
                                  contributions_file=contributions_file,
                                  distributions_file=distributions_file,
                                  topic=topic,
                                  tokenizer=self.tokenizer,
                                  context_length=context_length,
                                  policy=policy,
                                  sep_by_sent=sep_by_sent,
                                  for_inference=False,
                                  seed=seed)
        else:
            train_ds = torch.load(file)

        # define the training arguments
        training_args = TrainingArguments(output_dir="./results",
                                          num_train_epochs=epochs,
                                          per_device_train_batch_size=per_device_batch_size,
                                          per_device_eval_batch_size=per_device_batch_size,
                                          warmup_steps=0.1 * len(train_ds) / (per_device_batch_size * n_gpus),
                                          learning_rate=lr,
                                          weight_decay=weight_decay,
                                          logging_dir="./logs",
                                          logging_steps=1000,
                                          save_strategy="epoch",
                                          report_to="none",
                                          fp16=True)

        # define the trainer
        trainer = Trainer(model=self.model,
                          args=training_args,
                          train_dataset=train_ds)

        # start the training
        print("--- TRAINING THE MODEL ---")
        trainer.train()
        # save the model
        trainer.save_model(model_output_dir)
        self.model = trainer.model

    def get_highlights(self, 
                       sentences, 
                       context, 
                       n_hl=3
                       ):
        """
        This methods is used to extract some highlights from a collection of sentences and given a
        context.
        
        Parameters:
            sentence: list of sentences to be evaluated
            context: string containing the context
            n_hl: number of highlights to extract
        """
        if n_hl > len(sentences):
            n_hl = len(sentences)
        
        ds = InternalDataset(sentences=sentences,
                             context=context,
                             tokenizer=self.tokenizer)
        dl = DataLoader(ds, batch_size=4)
        
        print("--- EXTRACTING THE HIGHLIGHTS ---")        
        res = []
        for data in tqdm(dl):
            ids, am = data["input_ids"], data["attention_mask"]
            
            out = self.model(input_ids=ids.to(self.device), attention_mask=am.to(self.device))
            
            for t in out[0]:
                res.append(t.item())
                
        args = np.argsort(res)[::-1][:n_hl]
        return [sentences[i] for i in args]
    
    def get_highlights_on_dataset(self,
                                  file,
                                  contributions_file,
                                  distributions_file,
                                  topic,
                                  context_length=15,
                                  policy="random",
                                  sep_by_sent=True,
                                  seed=None,
                                  n_hl=15,
                                  return_references=False
                                  ):
        """
        This method is used to compute the highlights given a dataset. The dataset can be either a
        JSON following the structure described by PCEDataset or a pytorch dataset (PCEDataset).
        
        Parameters:
            file: path to the JSON file containing the dataset
            contributions_file: path to the JSON file with the contributions of each section
            distributions_file: path to the JSON file with the distributions of each section
            topic: the topic of the papers in the dataset ("AI", "BIO", "CS")
            context_length: number of sentences to compose the dataset
            policy: the in-bin choice strategy ("random", "rouge-2", "best")
            sep_by_sent: flag to decide whether or not to separate each sentence
            seed: seed for reproducibility of the dataset
            n_hl: number of highlights to be return for each paper
            return_references: flag to decide whether or not return the references
        """
        
        if file.split(".")[-1] == "json":
            ds = PCEDataset(file=file,
                            contributions_file=contributions_file,
                            distributions_file=distributions_file,
                            topic=topic,
                            tokenizer=self.tokenizer,
                            context_length=context_length,
                            policy=policy,
                            sep_by_sent=sep_by_sent,
                            for_inference=True,
                            seed=seed)
        else:
            ds = torch.load(file)
            
        dl = DataLoader(ds, batch_size=10)
        
        print("--- EXTRACTING HIGHLIGHTS ---")
        results = []
        for data in tqdm(dl):
            ids, am = data["input_ids"], data["attention_mask"]
            
            out = self.model(input_ids=ids.to("cuda"), attention_mask=am.to("cuda"))
            
            for tensor in out[0]:
                results.append(tensor.item())
        
        res = list(zip(results, ds.x, ds.paper_ids))
        map_ids_results = defaultdict(lambda: ([], []))
        for tup in res:
            map_ids_results[tup[2]][0].append(tup[0])  # the predicted rouge score
            map_ids_results[tup[2]][1].append(tup[1])  # the associated sentence
            
        res = dict()
        for k, v in map_ids_results.items():
            ixs = np.argsort(v[0])[::-1]
            best = [v[1][i] for i in ixs]

            res[k] = best[:n_hl]
            
        if return_references:
            return res, ds.get_highlights()
        else:
            return res
        
    def direct_extraction_on_dataset(self,
                                     file,
                                     contributions_file, 
                                     distributions_file,
                                     topic,
                                     return_references,
                                     seed=None,
                                     n_hl=15
                                     ):
        """
        This method performs direct extraction following sections' contributions and distributions.
        The JSON file should follow the structure defined by PCEDataset.
        
        Parameters:
            file: path to the JSON file containing the dataset
            contributions_file: path to the JSON file with the contributions of each section
            distributions_file: path to the JSON file with the distributions of each section
            topic: the topic of the papers in the dataset ("AI", "BIO", "CS")
            return_references: flag to decide whether or not return the references
            n_hl: number of highlights to be return for each paper
        """
        
        with open(file) as f:
            data = json.load(f)
        with open(contributions_file) as f:
            contrib = json.load(f)
            contrib = contrib[topic]
            contrib = {k: round(n_hl * v) for k, v in contrib.items()}
        with open(distributions_file) as f:
            prob = json.load(f)
            prob = prob[topic]
            
        rg = Rouge(metrics=["rouge-n", "rouge-l"], 
                   limit_length=False, 
                   max_n=2, 
                   alpha=0.5, 
                   stemming=False)
        
        if seed is not None:
            np.random.seed(seed)
        
        res = dict()
        references = dict()
        for pid, paper in tqdm(data.items()):
            if return_references:
                references[pid] = paper["highlights"]
                
            res[pid] = []
                
            for sec in ["introduction", "methods", "results", "discussion"]:
                if sec not in paper["sections"].keys() or len(paper["sections"][sec]) == 0:
                    continue
                
                bins = PCEDataset.define_bins(len(paper["sections"][sec]), len(prob[sec]))
                
                selected = []
                if len(paper["sections"][sec]) <= contrib[sec]:
                    selected = [i for i in range(len(paper["sections"][sec]))]
                else:
                    while len(selected) < contrib[sec]:
                        sel_bin = np.random.choice(len(prob[sec]), p=prob[sec])
                        if len(bins[sel_bin]) > 0:
                            ix = np.random.choice(len(bins[sel_bin]))
                            if len(paper["sections"][sec][ix]) > 1:
                                selected.append(bins[sel_bin][ix])
                            del bins[sel_bin][ix]
                            
                selected = [paper["sections"][sec][i] for i in selected]
                res[pid] = list(chain(res[pid], selected))
                
            rgs = [rg.get_scores(sent, paper["abstract"])["rouge-2"]["f"] for sent in res[pid]]
            args = np.argsort(rgs)[::-1]
            
            res[pid] = [res[pid][i] for i in args]
            
        if return_references:
            return res, references
        else:
            return res
    
    def evaluate_results(self,
                         results,
                         references=None,
                         n_hl=3
                         ):
        """
        This method can be used to perform the evaluation given the results of an extraction. The
        results can be either in disctionaries  or i a file followinf this structure:
        {
            "paper_id": {
                "system": [<string>, ...],
                "references": [<string>, ...]
            },
            ...
        }
        
        Parameters:
            results: either a JSON file as described above or a dictionary returned by this class
            references: dictionary return by this class, mandatory for results returned by this class
            n_hl: number of highlights on which to perform the evaluation
        """
        
        if isinstance(results, str):
            with open(results) as f:
                res = json.load(f)
        elif references is None:
            return "The references cannot be empty with this kind f results."
        else:
            res = defaultdict(lambda: dict())
            for k in results.keys():
                res[k]["system"] = results[k]
                res[k]["references"] = references[k]
        
        rg = Rouge(metrics=["rouge-n", "rouge-l"], 
                   limit_length=False, 
                   max_n=2, 
                   alpha=0.5, 
                   stemming=False)
                
        r1, r2, rl = [], [], []
        for k in results:
            sys = "\n".join(res[k]["system"][:n_hl])
            ref = "\n".join(res[k]["references"])
            
            ro = rg.get_scores(sys, ref)
            r1.append(ro["rouge-1"]["f"])
            r2.append(ro["rouge-2"]["f"])
            rl.append(ro["rouge-l"]["f"])
            
        return {f"R1@{n_hl}hl": np.mean(r1),
                f"R2@{n_hl}hl": np.mean(r2),
                f"RL@{n_hl}hl": np.mean(rl)}
                
    
    def evaluate_on_dataset(self,
                            file,
                            contributions_file,
                            distributions_file,
                            topic,
                            context_length=15,
                            policy="random",
                            sep_by_sent=True,
                            seed=None,
                            n_hl=3,
                            direct=False):
        """
        This methods is used to evaluate the model on a dataset. The dataset can be either a JSON
        following the format described by PCEDataset or a pytorch dataset (PCEDataset). If the
        dataset is a JSON this function can also perform direct extraction.
        
        Parameters:
            file: path to the JSON file containing the dataset
            contributions_file: path to the JSON file with the contributions of each section
            distributions_file: path to the JSON file with the distributions of each section
            topic: the topic of the papers in the dataset ("AI", "BIO", "CS")
            context_length: number of sentences to compose the dataset
            policy: the in-bin choice strategy ("random", "rouge-2", "best")
            sep_by_sent: flag to decide whether or not to separate each sentence
            seed: seed for reproducibility of the dataset
            n_hl: number of highlights to be return for each paper
            direct: flag to decide whether or not to perform direct extraction
        """
        
        if direct:
            if not file.split(".")[-1] == "json":
                return "The received file is not compatible with direct extraction."
            
            results, references = self.direct_extraction_on_dataset(file=file,
                                                                    contributions_file=contributions_file,
                                                                    distributions_file=distributions_file,
                                                                    topic=topic,
                                                                    return_references=True)
        else:
            results, references = self.get_highlights_on_dataset(file=file,
                                                                 contributions_file=contributions_file,
                                                                 distributions_file=distributions_file,
                                                                 topic=topic,
                                                                 context_length=context_length,
                                                                 policy=policy,
                                                                 sep_by_sent=sep_by_sent,
                                                                 seed=seed,
                                                                 n_hl=n_hl,
                                                                 return_references=True)
            
        return self.evaluate_results(results=results,
                                     references=references,
                                     n_hl=n_hl)
