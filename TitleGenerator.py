# Code for extension 1: generation of paper's title given the highlights and the abstract.

import torch
import evaluate
import json
import numpy as np

from tqdm import tqdm
from rouge import Rouge
from torch.utils.data import random_split, DataLoader
from transformers import BartForConditionalGeneration, AutoTokenizer, TrainingArguments, Trainer

from utils.reproducibility import *
from utils.datasets import TitleGenDataset


class TitleGenerator():
    def __init__(self, model_name="pietrocagnasso/bart-paper-titles"):
        """
        The models we fine-tuned for this extensions are available on huggingface:
        - pietrocagnasso/bart-paper-titles: fine-tuned for 1 epoch on all the papers in CS, AI,
                and BIO datasets
        - pietrocagnasso/bart-paper-titles-ai: starting from the general one this model is
                fine-tuned for an additional epoch on the AI dataset
        - pietrocagnasso/bart-paper-titles-bio: starting from the general one this model is
                fine-tuned for an additional epoch on the BIO dataset
        - pietrocagnasso/bart-paper-titles-cs: starting from the general one this model is
                fine-tuned for an additional epoch on the CS dataset

        Parameters:
            model_name: string with the name of the model to be used, by default it is the model
                trained on all the datasets
        """

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def train(self, train_ds_json, test_ds_json,
              use_highlights=True,
              use_abstract=True,
              epochs=1,
              per_device_batch_size=4,
              n_gpus=1,
              lr=5e-5,
              weight_decay=1e-2,
              model_output_dir="./best_model",
              seed=None
              ):
        """
        This method can be used to tune a model starting from scratch or fine-tuning a pre-trained
        model. For more than 1 training epochs it will use an 80/20 split to define the evaluation
        set, otherwise it will use the test set.
        Train and test sets are expected to be JSON files following the format required by
        TitleGenDataset.

        Parameters:
            train_ds_json: path to a json file following the format described above
            test_ds_json: path to a json file following the format described above
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
        train_ds = TitleGenDataset(json_file=train_ds_json,
                                   tokenizer=self.tokenizer,
                                   use_highlights=use_highlights,
                                   use_abstract=use_abstract)
        if epochs > 1:
            train_ds, eval_ds = random_split(train_ds,
                                        [int(0.8 * len(train_ds)), 
                                         len(train_ds) - int(0.8 * len(train_ds))])

        test_ds = TitleGenDataset(json_file=test_ds_json,
                                  tokenizer=self.tokenizer,
                                  use_highlights=use_highlights,
                                  use_abstract=use_abstract)
        if epochs == 1:
            eval_ds = test_ds

        # define the training arguments
        training_args = TrainingArguments(output_dir="./results",
                                          num_train_epochs=epochs,
                                          per_device_train_batch_size=per_device_batch_size,
                                          per_device_eval_batch_size=per_device_batch_size,
                                          warmup_steps=0.1 * len(train_ds) / (per_device_batch_size * n_gpus),
                                          learning_rate=lr,
                                          weight_decay=weight_decay,
                                          logging_dir="./logs",
                                          logging_steps=100,
                                          evaluation_strategy="epoch",
                                          save_strategy="epoch",
                                          load_best_model_at_end=True,
                                          metric_for_best_model="bertscore",
                                          report_to="none")

        # define the trainer
        trainer = Trainer(model=self.model,
                          args=training_args,
                          train_dataset=train_ds,
                          eval_dataset=eval_ds,
                          compute_metrics=self._compute_metric,
                          preprocess_logits_for_metrics=self._preprocess_logits_for_metrics)

        # start the training
        print("--- TRAIN THE MODEL ---")
        trainer.train()
        # save the model
        trainer.save_model(model_output_dir)
        self.model = trainer.model

        # test the model
        print("--- TEST THE MODEL ---")
        rg = Rouge(metrics=['rouge-n', 'rouge-l'],
                   limit_length=False, max_n=2,
                   alpha=0.5, stemming=False,
                   apply_best=False, apply_avg=True)
        bs = evaluate.load("bertscore")
        
        rouge1, rouge2, rougel, bertscore = 0, 0, 0, 0
        
        test_dl = DataLoader(test_ds, batch_size=8)

        for data in tqdm(test_dl):
            input_ids, labels = data["input_ids"], data["labels"]

            # decode the original titles
            real_titles = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # predict and decode titles
            outs = self.model.generate(input_ids.to(self.device),
                                       num_beams=5,
                                       min_length=3,
                                       max_length=32)
            pred_titles = self.tokenizer.batch_decode(outs, skip_special_tokens=True)

            # compute and update metrics
            rg_out = rg.get_scores(pred_titles, real_titles)
            rouge1 += rg_out["rouge-1"]["f"] * len(pred_titles)
            rouge2 += rg_out["rouge-2"]["f"] * len(pred_titles)
            rougel += rg_out["rouge-l"]["f"] * len(pred_titles)
            bs_res = bs.compute(predictions=pred_titles,
                                references=real_titles,
                                lang="en")
            bertscore += sum(bs_res["f1"])

        # compute the average of the metrics
        rouge1 /= len(test_ds)
        rouge2 /= len(test_ds)
        rougel /= len(test_ds)
        bertscore /= len(test_ds)

        print(f"""RESULTS:
        F1@rouge1: {rouge1}
        F1@rouge2: {rouge2}
        F1@rougel: {rougel}
        F1@bertscore: {bertscore}""")

    def generate_title(self,
                       highlights=None,
                       abstract=None,
                       use_highlights=True,
                       use_abstract=True
                       ):
        """
        This method can be used to compute the title given the highlights and the abstract of a
        single paper.

        Parameters:
            highlights: list of highlights for the paper
            abstract: abstract of the paper
            use_highlights: flag to trigger usage of highlights
            use_abstract: flag to trigger usage of the abstract
        """
        
        if (not use_highlights and not use_abstract) or (highlights is None and abstract is None):
            print("Warning: not elements to put inside the input.")
            return

        # compose the sentence
        s = f"{self.tokenizer.bos_token} "
        if use_highlights and highlights is not None:
            s += f" {self.tokenizer.sep_token} ".join(highlights)
        if use_abstract and abstract is not None:
            s += f" {self.tokenizer.sep_token} " + abstract + f" {self.tokenizer.eos_token}"

        # tokenize the sentence
        x = self.tokenizer.encode_plus(s,
                padding="max_length",
                max_length=1024,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        input_ids = x["input_ids"][0]

        # predict the title
        outs = self.model.generate(input_ids.unsqueeze(dim=0).to(self.device),
                                   num_beams=5,
                                   min_length=3,
                                   max_length=32)
        pred_title = self.tokenizer.decode(outs[0], skip_special_tokens=True)

        return pred_title

    def generate_titles(self, json_file,
                        use_highlights=True,
                        use_abstract=True,
                        batch_size=8
                        ):
        """
        This method can be used to predict the titles for a set of paper. The papers have to passed
        via a JSON file in the format required by TitleGenDataset.

        Parameters:
            json_file: the path to a JSON file following the format described above
            use_highlights: flag to trigger usage of highlights
            use_abstract: flag to trigger usage of the abstract
            batch_size: batch size to adapt to gpu memory limitations
        """

        # build the dataset
        ds = TitleGenDataset(json_file=json_file,
                    tokenizer=self.tokenizer,
                    use_highlights=use_highlights,
                    use_abstract=use_abstract,
                    inference=True
                )
        
        dl = DataLoader(ds, batch_size=batch_size)

        pred_titles = []

        print("--- PREDICTING TITLES ---")
        for data in tqdm(dl):
            input_ids = data["input_ids"]

            # predict and decode titles
            outs = self.model.generate(input_ids.to(self.device),
                                       num_beams=5,
                                       min_length=3,
                                       max_length=32)
            predictions = self.tokenizer.batch_decode(outs, skip_special_tokens=True)

            for title in predictions:
                pred_titles.append(title)

        return pred_titles
    
    def evaluate_on_dataset(self, json_file,
                            use_highlights=True,
                            use_abstract=True,
                            batch_size=8
                            ):
        """
        Methods that can be used to evaluate the metrics Rouge1, ROuge2, and BertScore for
        all the papers in a dataset.
        
        Parameters:
            json_file: path to a JSON file with the format required by the dataset
            use_highlights: flag to trigger usage of highlights
            use_abstract: flag to trigger usage of the abstract
        """
        rg = Rouge(metrics=['rouge-n', 'rouge-l'],
                   limit_length=False, max_n=2,
                   alpha=0.5, stemming=False,
                   apply_best=False, apply_avg=True)
        bs = evaluate.load("bertscore")
        
        with open(json_file) as f:
            data = json.load(f)
            
        references = [v["title"] for v in data.values()]
        predicted = self.generate_titles(json_file, use_highlights, use_abstract, batch_size)
        
        rg_out = rg.get_scores(predicted, references)
        rouge1 = rg_out["rouge-1"]["f"]
        rouge2 = rg_out["rouge-2"]["f"]
        rougel = rg_out["rouge-l"]["f"]
        bs_res = bs.compute(predictions=predicted,
                            references=references,
                            lang="en")
        bs = np.mean(bs_res["f1"])
        
        return {"F@rouge1": rouge1, "F@rouge2": rouge2, "F@rougel": rougel, "F@bertscore": bs}
    
    def _preprocess_logits_for_metrics(self, logits, labels):
        """
        This method is used internally to preprocess the logits so that they enter the metric
        computation method already as predictions.

        Parameters:
            logits: logits coming from the forward step of the model
            labels: labels associated to the logits
        """

        pred_ids = torch.argmax(logits[0], dim=-1)
        return pred_ids, labels

    def _compute_metric(self, pred):
        """
        This method is used internally to compute some metric in the training phase to decide the
        best model.

        Parameters:
            pred: predictions coming from the _preprocess_logits_for_metrics
        """
        rg = Rouge(metrics=['rouge-n', 'rouge-l'],
                   limit_length=False, max_n=2,
                   alpha=0.5, stemming=False,
                   apply_best=False, apply_avg=True)
        bs = evaluate.load("bertscore")

        # extract labels and predictions
        label_ids = pred.label_ids
        pred_ids = pred.predictions[0]

        # decode the ids
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        #compute the metrics
        rg_out = rg.get_scores(pred_str, label_str)
        bs_res = bs.compute(predictions=pred_str,
                            references=label_str,
                            lang="en")

        return {"bertscore": round(np.mean(bs_res["f1"]), 4),
                    "R1": round(rg_out["rouge-1"]["f"], 4),
                    "R2": round(rg_out["rouge-2"]["f"], 4),
                    "RL": round(rg_out["rouge-l"]["f"], 4)
                }
