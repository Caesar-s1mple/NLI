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
            tokens = self.tokenizer("[CLS] " + premise + " [SEP] " + hypothesis + " [SEP]",
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
    relation_list = []
    head_entities = []
    tail_entities = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            json_line = json.loads(line.strip())
            premise_list.append(json_line['text'])
            hypothesis_list.append(verbalizer[json_line['relation']].format(subj=json_line['h']['name'], obj=json_line['t']['name']))
            label_list.append("ENTAILMENT")
            relation_list.append(json_line['relation'])
            head_entities.append(json_line['h'])
            tail_entities.append(json_line['t'])

    return premise_list, hypothesis_list, label_list, relation_list, head_entities, tail_entities


verbalizer = {
    "/people/person/nationality": "{subj}'s nationality is {obj} .",
    "/time/event/locations": "{subj} happens in {obj} .",
    "/people/person/children": "{obj} is the children of {subj}",
    "/business/company/advisors": "{obj} is the advisors of {subj} .",
    "/business/location": "{obj} is the location of {subj} .",
    "/business/company/majorshareholders": "{obj} is the major shareholders of {subj}",
    "/people/person/place_lived": "{subj} lives in {obj}",
    "NA": "There's no relationship between {obj} and {subj} .",
    "/business/company/place_founded": "{obj} is the founded place of {subj} .",
    "/location/neighborhood/neighborhood_of": "{obj} is the neighborhood of {subj} .",
    "/people/deceasedperson/place_of_death": "{obj} is the place of death of {subj} .",
    "/film/film/featured_film_locations": "{obj} is the featured film location of {subj} .",
    "/location/region/capital": "{obj} is the capital of {subj} .",
    "/business/company/founders": "{obj} is the founder of {subj} .",
    "/people/ethnicity/geographic_distribution": "{obj} is the geographic distribution of {subj} .",
    "/location/country/administrative_divisions": "{subj} is the administrative division of {obj} .",
    "/people/deceasedperson/place_of_burial": "{obj} is the place of burial of {subj} .",
    "/location/country/capital": "{obj} is the capital of {subj} .",
    "/business/person/company": "{subj} works in {obj} .",
    "/location/location/contains": "{obj} is located in {subj} .",
    "/location/administrative_division/country": "{subj} is the administrative division of {obj} .",
    "/location/us_county/county_seat": "{obj} is the country seat of {subj} in America .",
    "/people/person/religion": "{subj} believes in {obj} .",
    "/people/person/place_of_birth": "{obj} is the birthplace of {subj} .",
    "/people/person/ethnicity": "{obj} is the ethnicity of {subj} ."
}

