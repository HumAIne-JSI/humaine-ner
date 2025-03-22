"""
These functions have been made for experimenting with Named Entity Recognition (NER) model called GLiNER. 
I was provided with a dataset from IJS E3 sector (sector for atrifficial intelligence), which consisted examples from multitude of different languages.
Main focus of experiment was to fine tune GLiNER base model on examples in Slovene and see how fast did the performance of the model improve.
In the E3 dataset there are 2971 examples to train on, but only 190 are in SLovene and those were relevant to this experiment. In the following code, there 
are functions for data modification to be appropriate for GLiNER, fine tuning functions and evaluation functions. There are also some more advanced 
functions which were not made by me and I have to thank to my coworkers who are much more skilled in programming than me for providing me with them.

Link to GLiNER on HuggingFace: https://huggingface.co/urchade/gliner_base
Link to E3 dataset for NER:    https://huggingface.co/E3-JSI
"""

# FIRSTLY, THE IMPORTS:

import json
from argparse import ArgumentParser
from pathlib import Path
import re
import matplotlib.pyplot as plt

import torch
from gliner import GLiNER
from tqdm import tqdm

import numpy as np
from typing_extensions import List, Literal, Tuple, TypedDict, Union

import random
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset

"""
These were the main imports, now to the functions. Next few functions were provided to me by a coworker and are intended for evaluating whether or not the 
true labels in an example match with the predicted labels from the model (GLiNER). As we will see, there are 3 different techniques for doing that: relaxed, exact and overlap.
"""

class Entity(TypedDict):
    text: str
    label: str

def compute_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score.

    Args:
        tp: Number of true positives.
        fp: Number of false pos

    Returns:
        The precision, recall, and F1 score.
    """
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0

    return p, r, f1

def longest_common_substring(text1: str, text2: str) -> float:
    """Find the longest common substring between two strings.

    Args:
        text1: The first string.
        text2: The second string.

    Returns:
        The length of the longest common substring.
    """

    m = len(text1)
    n = len(text2)
    lcsuff = np.zeros((m + 1, n + 1))
    result = 0  # To store length of the longest common substring

    # Building the lcsuff table in bottom-up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                lcsuff[i][j] = 0
            elif text1[i - 1] == text2[j - 1]:
                lcsuff[i][j] = lcsuff[i - 1][j - 1] + 1
                result = max(result, lcsuff[i][j])
            else:
                lcsuff[i][j] = 0

    return result

def _exact_ner_evaluation(
    true_ents: List[Entity], pred_ents: List[Entity]
) -> Tuple[float, float, float]:
    """Evaluate named entity recognition performance using exact matching.

    Args:
        true_ents: List of true entities.
        pred_ents: List of predicted entities.

    Returns:
        The true positives, false positives, and false negatives.
    """
    if len(true_ents) == 0 and len(pred_ents) == 0:
        return 1, 0, 0

    true_ents_set = set((ent["text"], ent["label"]) for ent in true_ents)
    pred_ents_set = set((ent["text"], ent["label"]) for ent in pred_ents)

    tp = len(true_ents_set & pred_ents_set)
    fp = len(pred_ents_set - true_ents_set)
    fn = len(true_ents_set - pred_ents_set)

    return tp, fp, fn

def _relaxed_ner_evaluation(
    true_ents: List[Entity], pred_ents: List[Entity]
) -> Tuple[float, float, float]:
    """Evaluate named entity recognition performance using relaxed matching.

    When using relaxed matching, the algorithm considers an entity as correct if the
    predicted entity contains the true entity. The labels of both entities must also match.

    Args:
        true_ents: List of true entities.
        pred_ents: List of predicted entities.

    Returns:
        The true positives, false positives, and false negatives.
    """
    if len(true_ents) == 0 and len(pred_ents) == 0:
        return 1, 0, 0

    true_ents_set = set((ent["text"], ent["label"]) for ent in true_ents)
    pred_ents_set = set((ent["text"], ent["label"]) for ent in pred_ents)

    tp = 0
    for true_ent in true_ents_set:
        for pred_ent in pred_ents_set:
            if true_ent[1] == pred_ent[1] and true_ent[0] in pred_ent[0]:
                tp += 1
                break

    fp = len(pred_ents_set) - tp
    fn = len(true_ents_set) - tp
    return tp, fp, fn

def _overlap_ner_evaluation(
    true_ents: List[Entity], pred_ents: List[Entity]
) -> Tuple[float, float, float]:
    """Evaluate named entity recognition performance using overlap matching.

    It is based on the longest common substring between the true and predicted entities.
    Furthermore, the label of the entity is also considered during the evaluation. The
    inspiration for this algorithm comes from the following paper:

    @inproceedings{bert-score,
        title={BERTScore: Evaluating Text Generation with BERT},
        author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
        booktitle={International Conference on Learning Representations},
        year={2020},
        url={https://openreview.net/forum?id=SkeHuCVFDr}
    }

    Args:
        true_ents: List of true entities.
        pred_ents: List of predicted entities.

    Returns:
        The precision, recall, and F1 score.
    """

    if len(true_ents) == 0 and len(pred_ents) == 0:
        return 1.0, 1.0, 1.0
    if len(true_ents) == 0 or len(pred_ents) == 0:
        return 0.0, 0.0, 0.0

    text_matrix = np.zeros((len(true_ents), len(pred_ents)))
    label_matrix = np.zeros((len(true_ents), len(pred_ents)))

    for i, true_ent in enumerate(true_ents):
        for j, pred_ent in enumerate(pred_ents):
            label_matrix[i, j] = true_ent["label"] == pred_ent["label"]
            text_matrix[i, j] = longest_common_substring(
                true_ent["text"], pred_ent["text"]
            ) / len(true_ent["text"])

    matrix = text_matrix * label_matrix
    p = np.mean(np.max(matrix, axis=0))
    r = np.mean(np.max(matrix, axis=1))
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0

    return p, r, f1

def evaluate_ner_performance(
    true_ents: List[List[Entity]],
    pred_ents: List[List[Entity]],
    match_type: Union[
        Literal["exact"], Literal["relaxed"], Literal["overlap"]
    ] = "exact",
) -> Tuple[float, float, float]:
    """Evaluate named entity recognition performance.

    Args:
        true_ents: List of true entities for each example.
        pred_ents: List of predicted entities for each example.
        match_type: The evaluation method to use, either "exact", "relaxed" or "overlap".

    Returns:
        The precision, recall, and F1 score.
    """
    if len(true_ents) != len(pred_ents):
        raise ValueError("The number of true and predicted entities must be the same.")

    if match_type not in ["exact", "relaxed", "overlap"]:
        raise ValueError(f"Unknown match_type method: {match_type}")

    if match_type == "overlap":
        p, r, f1 = 0.0, 0.0, 0.0
        for true_ent, pred_ent in zip(true_ents, pred_ents):
            _p, _r, _f1 = _overlap_ner_evaluation(true_ent, pred_ent)
            p, r, f1 = p + _p, r + _r, f1 + _f1
        p, r, f1 = p / len(true_ents), r / len(true_ents), f1 / len(true_ents)
        return p, r, f1

    if match_type == "exact":
        eval_func = _exact_ner_evaluation
    elif match_type == "relaxed":
        eval_func = _relaxed_ner_evaluation

    tp, fp, fn = 0, 0, 0
    for true_ent, pred_ent in zip(true_ents, pred_ents):
        _tp, _fp, _fn = eval_func(true_ent, pred_ent)
        tp, fp, fn = tp + _tp, fp + _fp, fn + _fn

    p, r, f1 = compute_metrics(tp, fp, fn)
    return p, r, f1

def main(dataset, model, tr, cpu = True):
    """
    This function takes as an input a dataset and than for all 3 different types of evaluation it tries to evaluate it.

    input: [{'text': 'Barack Obama was born in Honolulu .',
  'labels': [{'text': 'Barack Obama', 'label': 'PERSON'},
   {'text': 'Honolulu', 'label': 'LOCATION'}]}]

    output: dictionary with keywords "exact", "relaxed" and "overlap".
    """
    if not Path(dataset).exists():
        raise FileNotFoundError(f"Test data file {dataset} does not exist")

    with open(dataset, "r", encoding="utf8") as f:
        data = json.load(f)

        
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    
    model = GLiNER.from_pretrained(model, load_tokenizer=True, local_files_only=True)

    model.to(device)

    performances = {
        "exact": {"p": 0.0, "r": 0.0, "f1": 0.0},
        "relaxed": {"p": 0.0, "r": 0.0, "f1": 0.0},
        "overlap": {"p": 0.0, "r": 0.0, "f1": 0.0},
    }

    labels = list(set([e["label"] for e in data for e in e["labels"]]))

    true_ents = []
    pred_ents = []
    for example in tqdm(data, desc="Processing examples"):
        true_ents.append(
            [label for label in example["labels"] if label["label"] in labels]
        )
        with torch.no_grad():
            _pred_ents = model.predict_entities(
                example["text"], labels, threshold= tr
            )
        pred_ents.append(_pred_ents)

    for match_type in performances:
        p, r, f1 = evaluate_ner_performance(true_ents, pred_ents, match_type)
        performances[match_type]["p"] = float(p)
        performances[match_type]["r"] = float(r)
        performances[match_type]["f1"] = float(f1)

    return performances


"""
Now, the next few functions are only for transforming a dataset, so that it is compatible with GLiNER's native fine tuning function.
"""

def transform_to_numbers(tokens, labels):
    """
    A side function that will help us with the augment function.
    """
    ner = []
    for i, element in enumerate(tokens):
        if labels[i][0] == "B":
            sidelist = [i]
            if labels[i + 1][0] != "I":
                sidelist.append(i)
                sidelist.append(labels[i][2:])
                ner.append(sidelist)
                
        if labels[i][0] == "I":
            sidelist.append(i)
            if i == len(tokens) - 1:
                sidelist.append(labels[i][2:])
                ner.append(sidelist)
            
            if labels[i + 1][0] != "I":
                sidelist.append(labels[i][2:])
                ner.append(sidelist)
                
    return ner

def augment(data):
    """
    This function will transform a dataset in the form of B-label, I-label and 0 
    into GLiNER friendly form using list of lists such as [1, 2, "name"].

    THIS FORM IS NEEDED TO FINE TUNE GLiNER MODEL.

    input: dataset in form of [{'tokens': ['Barack', 'Obama', 'was', 'born', 'in', 'Honolulu', '.'],
    'labels': ['B-PERSON', 'I-PERSON', 'O', 'O', 'O', 'B-LOCATION', 'O']},
    {'tokens': ['Apple', 'Inc.', 'is', 'based', 'in', 'Cupertino', '.'],
    'labels': ['B-ORG', 'I-ORG', 'O', 'O', 'O', 'B-LOCATION', 'O']}]

    output: dataset in form of [{'tokenized_text': ['Barack', 'Obama', 'was', 'born', 'in', 'Honolulu', '.'], 
    'ner': [[0, 1, 'PERSON'], [5, 5, 'LOCATION']]}, 
    {'tokenized_text': ['Apple', 'Inc.', 'is', 'based', 'in', 'Cupertino', '.'], 
    'ner': [[0, 1, 'ORG'], [5, 5, 'LOCATION']]}]
    """
    result = []
    for i, element in enumerate(data):
        tokens = data[i]["tokens"]
        labels = data[i]["labels"]
        ner_ = transform_to_numbers(tokens, labels)
        trenutni = {}
        trenutni["tokenized_text"] = tokens
        trenutni["ner"] = ner_
        result.append(trenutni)
    return result

def pretvori(dataset):
    """
    This function transforms the data from form that has list of lists for labels into evaluation ready form

    THIS FORM IS NEEDED FOR EVALUATION OF GLINER

    input: {'tokenized_text': ['Barack', 'Obama', 'was', 'born', 'in', 'Honolulu', '.'], 'ner': [[0, 1, 'PERSON'], [5, 5, 'LOCATION']]}, 
    {'tokenized_text': ['Apple', 'Inc.', 'is', 'based', 'in', 'Cupertino', '.'], 'ner': [[0, 1, 'ORG'], [5, 5, 'LOCATION']]}]

    output: [{'text': 'Barack Obama was born in Honolulu .', 'labels': [{'text': 'Barack Obama', 'label': 'PERSON'},
   {'text': 'Honolulu', 'label': 'LOCATION'}]},
 {'text': 'Apple Inc. is based in Cupertino .',
  'labels': [{'text': 'Apple Inc.', 'label': 'ORG'},
   {'text': 'Cupertino', 'label': 'LOCATION'}]}]
    """
    besedilo = []
    for i, sentence in enumerate(dataset):
        stavek = {}
        s = " ".join(sentence["tokenized_text"])
        stavek["text"] = s
        stavek["labels"] = []
        for element in dataset[i]["ner"]:
            primer = {}
            start = element[0]
            end = element[1]
            ner = element[2]
            result = " ".join(dataset[i]["tokenized_text"][start:end + 1])
            primer["text"] = result
            primer["label"] = ner
            stavek["labels"].append(primer)
        
        besedilo.append(stavek)
    return besedilo

def convert_to_gliner_e3(dataset):
  """
  This functions can be used to transform given dataset of E3 so that it can be used to fine tune GLiNER.
  """
  dataset = dataset.to_pandas()
  dataset = dataset[["gliner_tokenized_text", "gliner_entities"]]
  dataset = dataset.rename(columns={"gliner_tokenized_text": "tokenized_text", "gliner_entities": "ner"})
  dataset["ner"] = dataset["ner"].apply(lambda x: json.loads(x))
  return dataset.to_dict(orient="records") 

def treniraj(model_to_train, train_dataset, model_save_name, steps = 40):

    """
    This function fine tunes a GLiNER model on a dataset in the form of: 
    input Dataset: [{'tokenized_text': ['Barack', 'Obama', 'was', 'born', 'in', 'Honolulu', '.'], 'ner': [[0, 1, 'PERSON'], [5, 5, 'LOCATION']]}, 
    {'tokenized_text': ['Apple', 'Inc.', 'is', 'based', 'in', 'Cupertino', '.'], 'ner': [[0, 1, 'ORG'], [5, 5, 'LOCATION']]}, 
    {'tokenized_text': ['The', 'Google', 'headquarters', 'are', 'located', 'in', 'Mountain', 'View', '.'], 'ner': [[1, 1, 'ORG'], [6, 7, 'LOCATION']]}, 
    {'tokenized_text': ['John', 'Doe', 'lives', 'in', 'New', 'York', '.'], 'ner': [[0, 1, 'PERSON'], [4, 5, 'LOCATION']]}]
    """
    
    num_steps = steps
    batch_size = 1
    data_size = len(train_dataset)
    num_batches = data_size // batch_size
    num_epochs = max(1, num_steps // num_batches)

    model = model_to_train
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)
    
    training_args = TrainingArguments(
        output_dir="models",
        learning_rate=5e-6,
        weight_decay=0.01,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear", #cosine
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2,
        num_train_epochs=num_epochs,
        evaluation_strategy="steps",
        save_steps = 100,
        save_total_limit=10,
        dataloader_num_workers = 0,
        use_cpu = False,
        report_to="none",
        )
    
    trainer = Trainer(
        model=model_to_train,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=[],
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    trainer.save_model(f"{model_save_name}")

"""
The next few functions are just some side functions that will be used later down in the code
"""

def pretvori_v_stavek(tokenised_text):
    stavek = tokenised_text
    sentence = ' '.join(stavek)
    sentence = sentence.replace(' ,', ',').replace(' .', '.').replace(' -', '-')
    return sentence

def add_to_dict(key, value, my_dict):
    if key not in my_dict:
        my_dict[key] = [value]
    else:
        my_dict[key].append(value)
    return my_dict

def compare_strings_ignore_spaces(a, b):
    # Define a pattern to remove spaces around common punctuation marks
    punctuation_pattern = r'\s*([,;:_\-])\s*'
    
    # Remove spaces around common punctuation marks
    a_cleaned = re.sub(punctuation_pattern, r'\1', a)
    b_cleaned = re.sub(punctuation_pattern, r'\1', b)
    
    # Also remove extra spaces anywhere in the string
    a_cleaned = " ".join(a_cleaned.split())
    b_cleaned = " ".join(b_cleaned.split())
    
    return a_cleaned == b_cleaned

"""
The next functions serves as the basis to evaluation of a model on a given example
"""
def primerjaj(example, model, technique = "exact", threshold = 0.9, show = False):
    """
    Ta funkcija pove presision, recall in f1 posameznega primera.
    
    Primer inputa: {'tokenized_text': array(['Dr.', 'Susan', 'Park', ',', 'a', 'cardiologist', 'at',
       'Riverside', 'Health', 'Center', ',', 'conducted', 'an', 'online',
       'consultation', 'on', 'February', '18', ',', '2023', ',', 'using',
       'the', 'IP', 'address', '192.168.1.10', '.'], dtype=object), 
       'ner': [[0, 2, 'person'], [5, 5, 'specialization'], [7, 9, 'hospital'], [16, 19, 'date'], [25, 25, 'ip address']]}

    Technique can be chosen from: exact, relaxed, relaxed_both and overlap
    
    If show == True, the function will show for that specific case number of True positives, false positives and false negatives,
    """
    besede_v_stavku = example["tokenized_text"]
    stavek_pravilni = pretvori_v_stavek(besede_v_stavku)
    if show == True:
        print(stavek_pravilni)
        print("\n")
    ners = example["ner"]
    true_events = []
    predicted_events = []

    
    for element in ners:
        a = " ".join(besede_v_stavku[element[0]:element[1]+1])
        if technique == "relaxed":
            a = a.replace(" ", "")
        true_events.append({"text":a, "label":element[2]})

    label_types = list(set([element[2] for element in ners]))
    entities = model.predict_entities(stavek_pravilni, label_types, threshold=threshold)
    for element in entities:
        b = element["text"]
        if technique == "relaxed":
            b = b.replace(" ", "")
        predicted_events.append({"text":b, "label":element["label"]})

    length = len(true_events)

    if technique == "exact":
        tp, fp, fn = _exact_ner_evaluation(true_events, predicted_events)
    elif technique == "relaxed":
        tp, fp, fn = _relaxed_ner_evaluation(true_events, predicted_events)
    elif technique == "relaxed_both":
        if _relaxed_ner_evaluation(true_events, predicted_events)[0] >= _relaxed_ner_evaluation(predicted_events, true_events)[0]:
            tp, fp, fn = _relaxed_ner_evaluation(true_events, predicted_events)
        else:
            tp, fp, fn = _relaxed_ner_evaluation(predicted_events, true_events)
    elif technique == "overlap":
        tp, fp, fn = _overlap_ner_evaluation(true_events, predicted_events)

    if show == True:
        print("TRUE VALUES:")
        for element in true_events:
            print(element)
        print("\n")
        print("PREDICTED VALUES:")
        if len(predicted_events) == 0:
            print("NO FOUNDINGS")
        else:
            for element in predicted_events:
                print(element)

    if tp + fp == 0 and tp + fn != 0:
        p = 0
        r = tp/(tp + fn)
        if r == 0:
            f1 = 0
            if show == True:
                print("presicion = 0, recall = 0, therefore f1 = 0")
        else:
            f1 = 2*((p*r)/(p+r))
            if show == True:
                print("precision = 0")
            
    elif tp + fp != 0  and tp + fn == 0:
        p = tp/(tp + fp)
        r = 0
        if p == 0:
            f1 = 0
            if show == True:
                print("presicion = 0, recall = 0, therefore f1 = 0")
        else:
            f1 = 2*((p*r)/(p+r))
            if show == True:
                print("recall = 0")
    elif tp + fp == 0 and tp + fn == 0:
        p, r, f1 = 0, 0, 0
        if show == True:
            print("presicion = 0, recall = 0, therefore f1 = 0")
    else:
        p, r = tp/(tp + fp), tp/(tp + fn)
        if p == 0 and r == 0:
            f1 = 0
        else:
            f1 = 2*((p*r)/(p+r))

    if show == True:
        print("\n")
        print(f"True positives: {tp}, False positives: {fp}, False negatives: {fn}")
    return p, r, f1

def performance_check(dataset, model,percentage_of_dataset = 0.1, threshold = 0.5):
    p, r, f1 = 0, 0, 0
    N = round(len(dataset)*percentage_of_dataset)
    
    for element in dataset[:N]:
        p_, r_, f1_ = primerjaj(element, model, threshold = threshold, show = False)
        p += p_
        r += r_
        f1 += f1_
    p = p/N
    r = r/N
    f1 = f1/N
    return p, r, f1

def tp_fp_fn(true, predict):
    '''
    Vrne True positives, false positives in false negatives.
    primer inputa:
    a = [{'text': 'Združenje zdravnikov', 'label': 'organizacija'},{'text': 'od ponedeljka do petka', 'label': 'delovni čas'},{'text': 'od 8:00 do 16:00 ure', 'label': 'delovni čas'}]
    b = [{'text': 'Združenje zdravnikov', 'label': 'organizacija', 'score': 0.8722029328346252}, {'text': 'ponedeljka do petka', 'label': 'delovni čas', 'score': 0.7550597190856934}]
    '''
    def clean_string(s):
        """Remove all spaces and standardize the string (optional: remove other unwanted characters like dashes)."""
        return re.sub(r'\s+', '', s)  # Remove all spaces
    
    def compare_labels(a, b):
        # Initialize lists
        true_positives = []
        false_positives = []
        false_negatives = []
    
        # Convert list `a` into a set of tuples (cleaned text, label) for easier comparison
        a_set = {(clean_string(item['text']), item['label']) for item in a}
    
        # Iterate through predictions in `b`
        for item_b in b:
            text_b = clean_string(item_b['text'])
            label_b = item_b['label']
            score_b = item_b['score']
    
            # Check if the prediction exists in `a`
            is_true_positive = False
    
            for text_a, label_a in a_set:
                if label_b == label_a and text_b in text_a:  # Check if text_b is a part of text_a
                    true_positives.append(item_b)
                    is_true_positive = True
                    break
    
            if not is_true_positive:
                false_positives.append(item_b)
    
        # Check for false negatives by comparing true labels in `a` that are missing in `b`
        a_text_labels = {(clean_string(item['text']), item['label']) for item in a}
        b_text_labels = {(clean_string(item['text']), item['label']) for item in b}
    
        for item_a in a:
            text_a = item_a['text']  # Keep the original text from `a`
            label_a = item_a['label']
            if (clean_string(text_a), label_a) not in b_text_labels:
                false_negatives.append({'text': text_a, 'label': label_a})
    
        return true_positives, false_positives, false_negatives
    return compare_labels(true,predict)

def examine(model, example,technique = "exact", threshold = 0.8, show = False):
    """
    Ta funkcija pove tp, fp in fn posameznega primera.
    
    Primer inputa: {'tokenized_text': array(['Dr.', 'Susan', 'Park', ',', 'a', 'cardiologist', 'at',
       'Riverside', 'Health', 'Center', ',', 'conducted', 'an', 'online',
       'consultation', 'on', 'February', '18', ',', '2023', ',', 'using',
       'the', 'IP', 'address', '192.168.1.10', '.'], dtype=object), 
       'ner': [[0, 2, 'person'], [5, 5, 'specialization'], [7, 9, 'hospital'], [16, 19, 'date'], [25, 25, 'ip address']]}

    Technique can be chosen from: exact, relaxed, relaxed_both and overlap
    
    If show == True, the function will show for that specific case number of True positives, false positives and false negatives,
    """
    besede_v_stavku = example["tokenized_text"]
    stavek_pravilni = pretvori_v_stavek(besede_v_stavku)
    if show == True:
        print(stavek_pravilni)
        print("\n")
    ners = example["ner"]
    true_events = []
    predicted_events = []

    
    for element in ners:
        a = " ".join(besede_v_stavku[element[0]:element[1]+1])
        if technique == "relaxed":
            a = a.replace(" ", "")
        true_events.append({"text":a, "label":element[2]})

    label_types = list(set([element[2] for element in ners]))
    entities = model.predict_entities(stavek_pravilni, label_types, threshold=threshold)
    for element in entities:
        b = element["text"]
        if technique == "relaxed":
            b = b.replace(" ", "")
        predicted_events.append({"text":b, "label":element["label"], "score":element["score"]})

    return tp_fp_fn(true_events, predicted_events)

def examine_prf1(model, example,technique = "exact", threshold = 0.8, show = False):
    true_positives, false_positives, false_negatives = examine(model, example,technique=technique, threshold = threshold, show = show)
    if show == True:
        print("TRUE POSITIVES:")
        for element in true_positives:
            print(element)
        print("\n")
        print("FALSE POSITIVES:")
        for element in false_positives:
            print(element)
        print("\n")
        print("FALSE NEGATIVES:")
        for element in false_negatives:
            print(element)
        print("\n")
    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    if tp + fp == 0 and tp + fn != 0:
        p = 0
        r = tp/(tp + fn)
        if r == 0:
            f1 = 0
            if show == True:
                print("presicion = 0, recall = 0, therefore f1 = 0")
        else:
            f1 = 2*((p*r)/(p+r))
            if show == True:
                print("precision = 0")
            
    elif tp + fp != 0  and tp + fn == 0:
        p = tp/(tp + fp)
        r = 0
        if p == 0:
            f1 = 0
            if show == True:
                print("presicion = 0, recall = 0, therefore f1 = 0")
        else:
            f1 = 2*((p*r)/(p+r))
            if show == True:
                print("recall = 0")
    elif tp + fp == 0 and tp + fn == 0:
        p, r, f1 = 0, 0, 0
        if show == True:
            print("presicion = 0, recall = 0, therefore f1 = 0")
    else:
        p, r = tp/(tp + fp), tp/(tp + fn)
        if p == 0 and r == 0:
            f1 = 0
        else:
            f1 = 2*((p*r)/(p+r))

    if show == True:
        print(f"True positives: {tp}, False positives: {fp}, False negatives: {fn}")
    return p, r, f1

def find_lowest_f1(model, left_dataset,trained_dataset = [], N = 10, threshold = 0.8, kriterij = "f1"):

    def update(a, b, indices):
        a = np.array(a)
        b = np.array(b)
        c = np.array(b[indices])
        b_new = np.delete(b, indices)
        a_new = np.append(a, c)
        return a_new, b_new, c

    f1_list = []
    if kriterij == "f1":
        a = 2
    elif kriterij == "p":
        a = 0
    elif kriterij == "r":
        a = 1

    for element in tqdm(left_dataset):
        f1 = examine_prf1(model, element, show = False, threshold = threshold)[a]
        f1_list.append(f1)
    
    f1_list = np.array(f1_list)
    indices = np.argsort(f1_list)[:N]

    new_trained_dataset, new_left_dataset, to_train = update(trained_dataset, left_dataset, indices)
    return new_trained_dataset, new_left_dataset, to_train, f1_list[indices]


