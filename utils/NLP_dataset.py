import json
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset

class EncodedDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=1024, include_highlights=True, include_abstract=True, is_test=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
        self.x = []
        self.y = []
        
        sep = " "
        
        # check that at least come data will be added to the content
        if not include_highlights and not include_abstract:
            print("!!!!! WARNING: must contein at least one between highlights and abstract. !!!!!")
            return
        
        # read the file with data
        with open(json_file) as f:
            data = json.load(f)
            
        for paper in data.values():
            s = tokenizer.bos_token
            
            s += sep.join(paper["highlights"]) if include_highlights else ""
            
            if include_abstract:
                if include_highlights:  # add separator between highlights and abstract
                    s += sep
                    
                # check if the entire abstract fits in the length limit of the model
                if len(s) + len(paper["abstract"]) <= max_length:
                    s += paper["abstract"]
                else:  # if not add sentence by sentence until reaching the length limit
                    for token in sent_tokenize(paper["abstract"]):
                        if len(s) + len(token) <= max_length:
                            s += token
                        else:
                            break
                            
            s += tokenizer.eos_token
                            
            self.x.append(s)
            self.y.append(paper["title"])
            
    def __getitem__(self, i):
        ins = self.tokenizer.encode_plus(self.x[i],
                                    padding="max_length",
                                    max_length=self.max_length,
                                    truncation=True,
                                    return_attention_mask=True,
                                    return_tensors='pt',
                                    return_token_type_ids=True)
        input_ids = ins["input_ids"]
        attention_mask = ins["attention_mask"]
        token_type_ids = ins["token_type_ids"]
        
        outs = self.tokenizer.encode_plus(self.x[i],
                                    self.y[i],
                                    padding="max_length",
                                    max_length=self.max_length,
                                    truncation=True,
                                    return_attention_mask=True,
                                    return_tensors='pt',
                                    return_token_type_ids=True)
        output_ids = outs["input_ids"]
        
        if self.is_test is None:
            return input_ids[0], attention_mask[0], token_type_ids[0]
        else:
            return input_ids[0], attention_mask[0], token_type_ids[0], output_ids[0]
    
    def __len__(self):
        return len(self.x)
    
    def _gettexts_(self, i):
        return self.x[i], self.y[i]
