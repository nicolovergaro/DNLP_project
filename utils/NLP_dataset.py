import json
from nltk.tokenize import sent_tokenize


class NLP_dataset(Dataset):
    def __init__(self, json_file, model_name="gpt2", max_length=1024, add_abstract=True, add_highlights=True, is_test=True):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        self.is_test = is_test
        self.papers = []
        self.labels = []
        
        end_token = "<|endoftext|>"
        start_token = "<|startoftext|>"
        sep = " "
        
        # check that at least some data will be put in the strings
        if not add_abstract and not add_highlights:
            print("Must contain at least one betwen highlights and abstract!")
            return
        
        # read the data from the json file
        with open(json_file) as f:
            data = json.load(f)
        
        # create a string that contains highlights (optional) and abstract (optional)
        for paper in data.values():
            s = start_token

            s += " ".join(paper["highlights"]) if add_highlights else ""
            if add_abstract:
                if add_highlights:
                    # add separator between highlights and abstract
                    s += sep
                abstract = paper['abstract']
                if len(s) + len(abstract) < max_length:
                    s+=paper['abstract']
                else:
                    # tokenize and add sentences until we reach a length of 1024
                    for token in sent_tokenize(abstract):
                        if len(s) + len(token) <= max_length - len(end_token):
                            s += token
                        else:
                            break
                            
            s += end_token
            self.papers.append(s)
            if self.is_test:
                self.labels.append(paper['title'])
    
    def __len__(self):
        return len(self.papers)
    
    def __getitem__(self,item):
        if self.is_test:  # test
            return self.papers[item]
        else:
            return self.papers[item], self.labels[item]
            
