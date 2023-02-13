import json
import numpy as np
from tqdm import tqdm
from rouge import Rouge
from collections import defaultdict
from torch.utils.data import Dataset


class TitleGenDataset(Dataset):
    def __init__(self,
                 json_file,
                 tokenizer,
                 max_input_length=1024,
                 max_target_length=128,
                 use_highlights=True,
                 use_abstract=True,
                 inference=False
                 ):
        """
        Dataset class for the title generation task. The expected file is a JSON structured as 
        follows:
        {
            "paper_id": {
                "highlights": [<string>, ...],
                "abstract": <string>,
                "title": <string>
            },
            ...
        }
        
        Parameters:
            json_file: path to a file containing the dataset
            tokenizer: chosen tokenizer for the task
            max_input_length: maximum input length supported by the chosen model
            max_target_length: maximum wanted length for the predicted text
            use_highlights: flag to use or not the highlights
            use_abstract: flag to use or not the abstract
            inference: flag to indicate that we need or nod the target label
        """
        
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.inference = inference
        self.inputs = []
        self.targets = []
        
        # read the data from the JSON
        with open(json_file) as f:
            data = json.load(f)

        # get the useful tokens
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        sep = tokenizer.sep_token

        for paper in data.values():
            s = bos  # add bos token

            # add highlights, if requested
            if use_highlights:
                s += " ".join(paper["highlights"])

            # add abstract, if requested
            if use_abstract:
                if use_highlights:  # add a separation between highlights and abstract
                    s += sep
                s += paper["abstract"]

            s += eos  # add eos token

            self.inputs.append(s)
            if not inference:
                self.targets.append(paper["title"])

    def __getitem__(self, index):
        # compute encodings for the input
        x = self.tokenizer.encode_plus(self.inputs[index],
                                       padding="max_length",
                                       max_length=self.max_input_length,
                                       truncation=True,
                                       return_attention_mask=True,
                                       return_tensors='pt')
        
        if self.inference:
            return {"input_ids": x["input_ids"][0], 
                    "attention_mask": x["attention_mask"][0]}
        else:
            # compute the encoding for the output
            y = self.tokenizer.encode_plus(self.targets[index],
                                           padding="max_length",
                                           max_length=self.max_target_length,
                                           truncation=True,
                                           return_tensors='pt')
            return {"input_ids": x["input_ids"][0], 
                    "attention_mask": x["attention_mask"][0], 
                    "labels": y["input_ids"][0]}

    def __len__(self):
        return len(self.inputs)


class PCEDataset(Dataset):
    def __init__(self, 
                 file, 
                 contributions_file, 
                 distributions_file, 
                 topic, 
                 tokenizer,
                 context_length=15,
                 policy="random",
                 sep_by_sent=True,
                 for_inference=False,
                 seed=None
                ):
        """
        Dataset class to build datasets compatible with the Probabilistic Context Extraction method.
        The expected input file is a JSON structured as follows:
        {
            "paper_id": {
                "highlights": [<string>, ...],
                "abstract": <string>,
                "sections": {
                    "abstract": [<string>, ...],
                    "introduction": [<string>, ...],
                    "methods": [<string>, ...],
                    "results": [<string>, ...],
                    "discussion": [<string>, ...]
                },
                "rouges": {
                    "abstract": [<float>, ...],
                    "introduction": [<float>, ...],
                    "methods": [<float>, ...],
                    "results": [<float>, ...],
                    "discussion": [<float>, ...]
                }
            },
            ...
        }
        The field rouges is mandatory only for training sets (for_inference=False), it contains the 
        highest Rouge-2 of the corresponding sentence wrt an highlight.
        The contributions file is expected to follow this structure:
        {
            "topic": {
                "abstract": <float>,
                "introduction": <float>,
                "methods: <float>,
                "results": <float>,
                "discussion": <float>
            },
            ...
        }
        and the distributions file is expected to follow this structure:
        {
            "topic": {
                "abstract": [<float>, ...],
                "introduction": [<float>, ...],
                "methods": [<float>, ...],
                "results": [<float>, ...],
                "discussion": [<float>, ...]
            },
            ...
        }
        
        Parameters:
            file: path to the JSON file containing the dataset
            contributions_file: path to the JSON file with the contributions of each section
            distributions_file: path to the JSON file with the distributions of each section
            topic: the topic of the papers in the dataset ("AI", "BIO", "CS")
            tokenizer: chosen tokenizer for the task
            context_length: number of sentences to compose the dataset
            policy: the in-bin choice strategy ("random", "rouge-2", "best")
            sep_by_sent: flag to decide whether or not to separate each sentence
            for_inference: flag to decide if the dataset is for inference
            seed: seed for reproducibility of the dataset
        """
        
        self.tokenizer = tokenizer
        self.for_inference = for_inference
        
        self.x = []
        self.contexts = dict()
        self.y = []
        self.highlights = dict()
        self.paper_ids = []
        
        # read the dataset
        with open(file) as f:
            data = json.load(f)
                    
        # compute relative positions the top n sentences, normalized rouge-2 wise
        with open(distributions_file) as f:
            distributions = json.load(f)
        distributions = distributions[topic]
                
        # compute section contributions
        with open(contributions_file) as f:
            contributions = json.load(f)
        contributions = contributions[topic]
        contributions = {sec: round(context_length * pcg) for sec, pcg in contributions.items()}
    
        # reproducibility   
        if seed is not None:
            np.random.seed(seed)        
        
        # compose the dataset
        for pid in tqdm(data.keys()):            
            self.highlights[pid] = data[pid]["highlights"]
            
            # compose the context
            context = ""
            for sec in ["abstract", "introduction", "methods", "results", "discussion"]:
                if sec in list(data[pid]["sections"].keys()):
                    sents = PCEDataset.extract_sentences(sents=data[pid]["sections"][sec],
                                                         n=contributions[sec],
                                                         probabilities=distributions[sec],
                                                         policy=policy,
                                                         abstract=data[pid]["abstract"])
                    if len(sents) > 0:
                        if len(context) > 0:
                            context += f" {self.tokenizer.sep_token} "
                        if sep_by_sent:
                            context += f" {self.tokenizer.sep_token} ".join(sents)
                        else:
                            context += " ".join(sents)
            self.contexts[pid] = context
            
            # build sentence and targets lists
            for sec in data[pid]["sections"].keys():
                if sec != "abstract":
                    for i, sent in enumerate(data[pid]["sections"][sec]):
                        self.paper_ids.append(pid)
                        self.x.append(sent)
                        if not self.for_inference:
                            self.y.append(data[pid]["rouges"][sec][i])           
                        
    def __getitem__(self, index):
        # compute encodings for the input
        x = self.tokenizer.encode_plus(self.x[index],
                                       self.contexts[self.paper_ids[index]],
                                       padding="max_length",
                                       truncation=True,
                                       max_length=384,
                                       return_attention_mask=True,
                                       return_tensors='pt')
        
        if self.for_inference:
            return {"input_ids": x["input_ids"][0],
                    "attention_mask": x["attention_mask"][0]}
        else:
            return {"input_ids": x["input_ids"][0],
                    "attention_mask": x["attention_mask"][0],
                    "labels": self.y[index]}
        
    def __len__(self):
        return len(self.x)
    
    def get_highlights(self):
        return self.highlights
    
    def extract_sentences(sents, n, probabilities, policy, abstract=None):
        if policy not in ["random", "rouge-2", "best"]:
            raise ValueError("The policy you chose is not supported. Try with 'random', 'rouge-2' or 'best'.")
            
        # trivial case
        if len(sents) <= n:
            return sents
        
        if policy in ["rouge-2", "best"]:
            rg = Rouge(metrics=["rouge-n"], limit_length=False, max_n=2, alpha=0.5, stemming=False)
            rouges = []
            # compute rouge-2 F of each sentence wrt the abstract
            for sent in sents:
                rouges.append(rg.get_scores(sent, abstract)["rouge-2"]["f"])
            rouges = np.array(rouges)
        
        # get bins
        n_bins = len(probabilities)
        bins = PCEDataset.define_bins(len(sents), n_bins)
        
        # one sentence per bin
        if len(sents) == n_bins:
            selected = np.argsort(probabilities)[::-1][:n]
        # more than one sentence per bin
        elif policy == "best":
            # return the n sentences with higher rouge
            selected = np.argsort(rouges)[::-1][:n]
        else:
            if policy == "rouge-2":
                # sort the sentences in the bin according to their rouge
                for b, ixs in bins.items():
                    rgs = rouges[ixs]
                    args = np.argsort(rgs)[::-1]
                    bins[b] = [ixs[i] for i in args]
                    
            selected = []
            while len(selected) < n:
                # select a bin according to the probabilities
                sel_bin = np.random.choice(n_bins, p=probabilities)
                if len(bins[sel_bin]) > 0:
                    # select a sentence in the bin according to the policy
                    if policy == "random":
                        ix = np.random.choice(len(bins[sel_bin]))
                        if len(sents[ix]) > 1:
                            selected.append(bins[sel_bin][ix])
                        del bins[sel_bin][ix]
                    else:
                        if len(sents[bins[sel_bin][0]]) > 1:
                            selected.append(bins[sel_bin][0])
                        del bins[sel_bin][0]
                        
        # reorder the selected sentences according to the appearance order in the original text
        sorted(selected)
        return [sents[i] for i in selected]
    
    def define_bins(length, n_bins):
        # create a map between the bins and the sentences
        bin_length = length / n_bins
        pos_bin_mapping = {i: int(i / bin_length) for i in range(length)}
        bin_pos_mapping = defaultdict(lambda: [])
        for pos, curr_bin in pos_bin_mapping.items():
            bin_pos_mapping[curr_bin].append(pos)
        return bin_pos_mapping
