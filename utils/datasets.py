import json
import evaluate
import numpy as np
from rouge import Rouge
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset


class TitleGenDataset(Dataset):
    def __init__(self, json_file, tokenizer,
                    max_input_length=1024,
                    max_target_length=128,
                    use_highlights=True,
                    use_abstract=True,
                    inference=False):
        """
        Dataset class for the title generation task.
        
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
                  return_tensors='pt'
              )
        
        if self.inference:
            return {"input_ids": x["input_ids"][0], "attention_mask": x["attention_mask"][0]}
        else:
            # compute the encoding for the output
            y = self.tokenizer.encode_plus(self.targets[index],
                      padding="max_length",
                      max_length=self.max_target_length,
                      truncation=True,
                      return_tensors='pt'
                  )
            return {"input_ids": x["input_ids"][0], "attention_mask": x["attention_mask"][0], "labels": y["input_ids"][0]}

    def __len__(self):
        return len(self.inputs)

    
class PCEDataset(Dataset):
    def __init__(self, papers_file,
                    tokenizer,
                    contributions_file,
                    probabilities_file, 
                    topic, 
                    n_sentences=10, 
                    strategy=None,
                    inference=False):
        """
        Dataset class used for Probabilistic context extraction task.
        The JSON files are expected in the following formats:
        papers_file:
        {
            "papers": {
                "<paper id>": {
                    "id": "<paper id>",
                    "abstract": ["<sentence>", ...],
                    "introduction": ["<sentence>", ...],
                    "methods": ["<sentence>", ...],
                    "results": ["<sentence>", ...],
                    "discussion": ["<sentence>", ...],
                    "highlights": ["<highlight>"]
                },
                ...
            }
        }
        contribution_file:
        {
            "<topic>": {
                "abstract": <float>,  // these floats must sum up to 1
                "introduction": <float>,
                "methods": <float>,
                "results": <float>,
                "discussion": <float>,
            },
            ...
        }
        probabilites_file:
        {
            "<topic>": {
                "abstract": [<float>, ...],  // the floats in each list must sum up to 1
                "introduction": [<float>, ...],
                "methods": [<float>, ...],
                "results": [<float>, ...],
                "discussion": [<float>, ...],
            },
            ...
        }
        
        Parameters:
            papers_file: JSON file containing the papers
            contribution_file: JSON file containing the section contribution pre each topic
            probabilities_file: JSON file with the probability to fall in a certain bin
            tokenizer: tokenizer chosen for the task
            topic: topic of the papers (CS, AI, BIO)
            n_enrichments: number of sentences to be added to the context
            strategy: choice strategy of the sentence inside the bin
        """
        
        self.tokenizer = tokenizer
        self.x = []
        self.contexts = []
        self.y = []
        self.inference = inference

        rg = Rouge(metrics=['rouge-n'], limit_length=False, max_n=2, alpha=0.5, stemming=False, apply_best=True, apply_avg=False)

        # check for errors in topic selection
        if topic not in ["AI", "BIO", "CS"]:
            print("ERROR: unknown topic. Try with AI, BIO or CS.")
            return

        # read contributions and probabilities file and extract the values useful for the specified topic
        with open(contributions_file) as f:
            contributions = json.load(f)
        contrib = {k: round(n_sentences*v) for k, v in contributions[topic].items()}

        with open(probabilities_file) as f:
            probabilities = json.load(f)
        prob = probabilities[topic]

        # read data and build the dataset
        with open(papers_file) as f:
            data = json.load(f)

            # for each paper
            for paper in data["papers"]:
                context = dict()
                # compute section contriution in the context
                for sec in contrib.keys():
                    context[sec] = PCEDataset.select_sentences(paper, sec, prob[sec], contrib[sec], strategy)
                # compose the context
                context = f" {tokenizer.sep_token} ".join([v for v in context.values()])

                # for each section
                for sec in contrib.keys():
                    # for each sentence in the section
                    for sent in paper[sec]:
                        try:
                            if not inference:
                                # compute the rouge-2 f
                                r2f = rg.get_scores(sent, paper["highlights"])["rouge-2"]["f"]                            
                                self.y.append(r2f)
                            
                            self.x.append(sent)
                            self.contexts.append(context)
                        except:
                            pass
                        

    def __getitem__(self, index):
        # compute encodings for the input
        x = self.tokenizer.encode_plus(self.x[index],
                  self.contexts[index],
                  padding="max_length",
                  truncation=True,
                  return_attention_mask=True,
                  return_tensors='pt'
              )
        
        if self.inference:
            return {"input_ids": x["input_ids"][0], "attention_mask": x["attention_mask"][0]}
        else:
            return {"input_ids": x["input_ids"][0], "attention_mask": x["attention_mask"][0], "labels": self.y[index]}


    def __len__(self):
        return len(self.x)

    
    @staticmethod
    def select_sentences(paper,
                         section,
                         prob, 
                         n_sents, 
                         strategy=None):
        """
        Method to select the sentences that will compose the context.
        
        Parameters:
            paper: dictionary with the sections composing the paper
            section: section from which we want to extract the sentences
            prob: probability distribution across 20 bins
            n_sents: number of sentences that will be extracted
            strategy: strategy followed to extract the sentences (None or "abstract")
        """
        # if empty return an empty string
        if len(paper[section]) == 0 or n_sents == 0:
            return ""

        # if it requires more sentence than available return the entire section
        if n_sents > len(paper[section]):
            return " ".join(paper["section"])

        # the selection strategy can be based on the semantic similarity with the abstract
        if strategy == "abstract":
            abstract = " ".join(paper["abstract"])
            bs = evaluate.load("bertscore")

        chosen = []

        # compute the number of sentences covered by each bin
        sent_per_bin = []
        bounds_per_bin = []
        while len(sent_per_bin) < len(p):
            lb = int((1 / len(p)) * len(sent_per_bin) * len(paper[section])) # index lower bound
            ub = min(int((1 / len(p)) * (len(sent_per_bin) + 1) * len(paper[section])), len(paper[section])) - 1 # index upper bound
            bounds_per_bin.append((lb, ub))
            sent_per_bin.append(ub - lb + 1)
        # extract bins until they are compatible with the availability of sentences in each bin
        while True:
            flag = True
            # extract the bins from which we will select the sentences
            bins = np.random.choice(range(len(prob)), n_sents, p=prob)
            # count repeptitions of each element
            reps = np.bincount(bins)
            for max_n, n in zip(sent_per_bin, reps):
                # check that the repetitions don't overcome the length of each bin
                if n > max_n:
                    flag = False
            if flag:
                break

        # choose the sentences
        while len(chosen) < n_sents:
            # get bins lower and upper bound in index
            lb, ub = bounds_per_bin[bins[len(chosen)]]
            # extract the index
            if strategy == None:
                # if no strategy was chosen choose at random in the bin
                ix = np.random.choice(range(lb, ub+1))
            elif strategy == "abstract":
                # compute the bertscore wrt the abstract, choose in order the best sentences
                bsf = bs.compute(paper[section][lb:ub+1], [abstract]*len(ub - lb + 1))["f1"]
                ixs = np.argsort(bsf)[::-1]
                for ix in ixs:
                    if ix not in chosen:
                        break
                ix += lb
            
            if ix not in chosen:
                chosen.append(ix)

        chosen.sort()
        return " ".join([paper[section][i] for i in chosen])
