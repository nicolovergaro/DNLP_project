import json
from torch.utils.data import Dataset


class TitleGenDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_input_length=1024, max_target_length=128, use_highlights=True, use_abstract=True, inference=False):
        """
        Dataset class for the title generation task.
        
        Attributes:
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
