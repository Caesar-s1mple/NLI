import json
from torch.utils.data import Dataset
import torch
from convert import label2id


class NLIDataset(Dataset):
    def __init__(self, premise, hypothesis, labels, tokenizer, max_len):
        self.premise = premise
        self.hypothesis = hypothesis
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset = self.process(premise, hypothesis, labels)

    def process(self, origin_premise, origin_hypothesis, origin_labels):
        sentences = []
        labels = []
        attention_masks = []
        for premise, hypothesis in zip(origin_premise, origin_hypothesis):
            tokens = self.tokenizer("[CLS] " + premise + "[SEP] " + hypothesis + "[SEP]",
                                    add_special_tokens=False,
                                    return_input_ids=True,
                                    return_attention_mask=True)

            tokens_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']

            tokens_ids = fill_padding(tokens_ids, self.max_len, 0)
            attention_mask = fill_padding(attention_mask, self.max_len, 0)

            sentences.append(tokens_ids)
            attention_masks.append(attention_mask)

        for label in origin_labels:
            label_ids = fill_padding([label2id[label]], 1, 0)
            labels.append(label_ids)

        return {
            'sentences': sentences,
            "attention_masks": attention_masks,
            "labels": labels
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return {
            'sentences': self.dataset['sentences'][item],
            'attention_masks': self.dataset['attention_masks'][item],
            'labels': self.dataset['labels'][item]
        }


def fill_padding(data, max_len, pad_id):
    if len(data) < max_len:
        pad_len = max_len - len(data)
        padding = [pad_id for _ in range(pad_len)]
        data = torch.tensor(data + padding)
    else:
        data = torch.tensor(data[: max_len])
    return data


def preprocess(file):
    premise_list = []
    hypothesis_list = []
    label_list = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            json_line = json.loads(line.strip())
            premise_list.append(json_line['text'])
            hypothesis_list.append(json_line['h']['name'] +
                                   "is the" +
                                   json_line['relation'] +
                                   "of" +
                                   json_line['t']['name'])
            label_list.append("ENTAILMENT")

    return premise_list, hypothesis_list, label_list


verbalizer = {
    "/people/person/nationality": "{obj} is the nationality of {subj} .",
    "/time/event/locations": "{subj} is located in {obj}",
    "/people/person/children": "{obj} is the children of {subj}",
    "/business/company/advisors": "",
    "/business/location": "{subj} is located in {obj} .",
    "/business/company/majorshareholders": "",
    "/people/person/place_lived": "",
    "NA": "{subj} and {obj} are not related .",
    "/business/company/place_founded": "",
    "/location/neighborhood/neighborhood_of": "",
    "/people/deceasedperson/place_of_death": "{subj} died in {obj} .",
    "/film/film/featured_film_locations": "",
    "/location/region/capital": "{obj} is the capatial of {subj} .",
    "/business/company/founders": "{subj} was founded by {obj} .",
    "/people/ethnicity/geographic_distribution": "",
    "/location/country/administrative_divisions": "",
    "/people/deceasedperson/place_of_burial": "{subj} was burried in {obj} .",
    "/location/country/capital": "{obj} is the capital of {subj} .",
    "/business/person/company": "",
    "/location/location/contains": "",
    "/location/administrative_division/country": "",
    "/location/us_county/county_seat": "",
    "/people/person/religion": "{obj} is the religion of {subj} .",
    "/people/person/place_of_birth": "{subj} was born in {obj} .",
    "/people/person/ethnicity": ""
}

