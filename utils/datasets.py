import json
from torch.utils.data import Dataset


class TitleGenDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_input_length=1024, max_target_length=128, use_highlights=True, use_abstract=True, is_test=False):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.is_test = is_test
        self.inputs = []
        self.targets = []
        
        with open(json_file) as f:
            data = json.load(f)

        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        sep = tokenizer.sep_token

        for paper in data.values():
            s = bos  # begin of sequence token

            # add highlights if requested
            if use_highlights:
                s += " ".join(paper["highlights"])

            # add abstract if requested
            if use_abstract:
                if use_highlights:  # add a separation between highlights and abstract
                    s += sep
                s += paper["abstract"]

            s += eos  # end of sequence token

            self.inputs.append(s)
            if not is_test:
                self.targets.append(paper["title"])

    def __getitem__(self, index):
        # compute encodings for the input and the output
        x = self.tokenizer.encode_plus(self.inputs[index],
                                  padding="max_length",
                                  max_length=self.max_input_length,
                                  truncation=True,
                                  return_attention_mask=True,
                                  return_tensors='pt')
        y = self.tokenizer.encode_plus(self.targets[index],
                                  padding="max_length",
                                  max_length=self.max_target_length,
                                  truncation=True,
                                  return_tensors='pt')
        
        if self.is_test:
            return {"input_ids": x["input_ids"][0], "attention_mask": x["attention_mask"][0]}
        else:
            return {"input_ids": x["input_ids"][0], "attention_mask": x["attention_mask"][0], "labels": y["input_ids"][0]}

    def __len__(self):
        return len(self.inputs)
