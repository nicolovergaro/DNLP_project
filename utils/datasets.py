import json
from torch.utils.data import Dataset
from rouge import Rouge
from nltk.tokenize import sent_tokenize

rg = Rouge()


class TitleGenDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_input_length=1024, max_target_length=128, use_highlights=True, use_abstract=True, inference=False):
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
        # compute encodings for the input and the output
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
    def __init__(self, papers_file, contributions_file, probabilities_file, tokenizer, topic, n_enrichments=5, strategy=None):
        """
        Dataset class used for Probabilisti context enrichment task.
        
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

        # read topic stats
        if topic not in ["CS", "AI", "BIO"]:
            print("ERROR: unkonwn topic. Try with CS, AI or BIO.")
            raise ValueError

        # read the contributions file and get contributions for the topic
        with open(contributions_file) as f:
            contributions = json.load(f)
        cntrbs = {sec: round(contributions[topic][sec] * n_enrichments) for sec in contributions[topic]}

        # read the probabilities file and get probabilities for the topic
        with open(probabilities_file) as f:
            probabilities = json.load(f)
        p = probabilities[topic]

        # read data from the file
        with open(papers_file) as f:
            data = json.load(f)

            for i, paper in enumerate(tqdm(data["papers"])):
                # compose the context
                context = paper["ABSTRACT"] + f" {tokenizer.sep_token} "
                for sec in cntrbs:
                    # find and add the enrichments to the context
                    context += f" {tokenizer.sep_token} ".join(PCEDataset.select_enrichments(paper, sec, p[sec], cntrbs[sec], strategy))

                # construct the dataset
                for sent in sent_tokenize(paper["FULL_TEXT"]):
                    try:
                        # compute the rouge 2 f-score, the target of the regression
                        scores = rg.get_scores([sent] * len(paper["HIGHLIGHTS"]), paper["HIGHLIGHTS"])
                        r2f = [s["rouge-2"]["f"] for s in scores]

                        self.x.append(f"{self.tokenizer.cls_token} " + sent)
                        self.contexts.append(context)
                        self.y.append(max(r2f))
                    except: pass


    def __getitem__(self, i):
        out = self.tokenizer.encode_plus(self.x[i],
                                        self.contexts[i],
                                        padding="max_length",
                                        max_length=512,
                                        truncation=True,
                                        return_attention_mask=True,
                                        return_tensors='pt',
                                        return_token_type_ids=True)
        
        return {"input_ids": out["input_ids"][0], "attention_mask": out["attention_mask"][0], "labels": self.y[i]}

    def __len__(self):
        return len(self.x)

    @staticmethod
    def select_enrichments(paper, section, p, cntrb, strategy):
        """
        Method to probabilistically extract the enrichments.
        
        Parameters:
            paper: the paper on which we are working
            section: section from which we are going to extract the enrichments (introduction, methods, results and discussion)
            p: bins probability
            cntrb: number of enrichment to be extracted
            strategy: strategy to choose among sentence in the same bin (TO BE IMPLEMENTED)
        """
        
        chosen = []  #chosen enrichments
        sentences = sent_tokenize(paper[section.upper()])  # section sentences
        if len(sentences) == 0 or cntrb == 0:
            return []

        # compute the number of sentences covered by each bin
        sent_per_bin = []
        bounds_per_bin = []
        while len(sent_per_bin) < len(p):
            lb = int((1 / len(p)) * len(sent_per_bin) * len(sentences)) # index lower bound
            ub = min(int((1 / len(p)) * (len(sent_per_bin) + 1) * len(sentences)), len(sentences)) - 1 # index upper bound
            bounds_per_bin.append((lb, ub))
            sent_per_bin.append(ub - lb + 1)
        # extract bins until they are compatible with the availability of sentences in each bin
        while True:
            flag = True
            # extract the bins from which we will select the sentences
            bins = np.random.choice(range(len(p)), cntrb, p=p)
            # count repeptitions of each element
            reps = np.bincount(bins)
            for max_n, n in zip(sent_per_bin, reps):
                # check that the repetitions don't overcome the length of each bin
                if n > max_n:
                    flag = False
            if flag:
                break
        
        while len(chosen) < cntrb:
            # get bins lower and upper bound in index
            lb, ub = bounds_per_bin[bins[len(chosen)]]
            # extract the index
            if strategy == None:
                ix = np.random.choice(range(lb, ub+1))
            else:
                print("Warning: use None as strategy")
                raise ValueError
            
            if ix not in chosen and len(sentences[ix]) > 10:
                chosen.append(ix)
                
        chosen = [sentences[ix] for ix in chosen]
        return chosen
