from data_process import preprocess, NLIDataset
from models import Deberta_NLI
from transformers import DebertaTokenizer, logging
from transformers.utils.notebook import format_time
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from metrics import score
import torch.nn as nn
import numpy as np
import torch
import time
import json
import logging as log

logging.set_verbosity_error()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def inferring(batch_size):
    print("-----Inferring!-----")
    premise, hypothesis, labels, relations, head_entities, tail_entities = preprocess('./dataset/nyt10m_train.txt')

    bag = {}
    for relation in relations:
        bag[relation] = []

    tokenizer = DebertaTokenizer.from_pretrained('./deberta-large-mnli')
    dataset = NLIDataset(premise, hypothesis, labels, tokenizer, 256)

    data_loader = DataLoader(dataset, batch_size=batch_size)
    print("-----Dataloader Build!-----")

    model = Deberta_NLI.from_pretrained('./deberta-large-mnli')
    model.to(device)

    f = open('./res/res.txt', 'w', encoding='utf-8')
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            sentences = batch['sentences'].to(device)
            attention_mask = batch['attention_masks'].to(device)

            outputs = model(sentences, attention_mask=attention_mask, return_dict=True)

            entailment_scores = torch.softmax(outputs['logits'], dim=1)[:, 2].cpu().numpy().astype(float)

            for i in range(len(batch['labels'])):
                f.write(json.dumps({
                    "premise": premise[step * batch_size + i],
                    "hypothesis": hypothesis[step * batch_size + i],
                    "score": entailment_scores[i],
                    "relation": relations[step * batch_size + i],
                    "h": head_entities[step * batch_size + i],
                    't': tail_entities[step * batch_size + i]
                }) + '\n')

            if step % 500 == 0:
                print("step: {}/{}   {}".format(step, len(data_loader), format_time(time.time() - t0)))

    f.close()

    print("-----Inferring Finished!-----")


def test(model=None, after_epoch=False):
    log.info("-----Testing!-----")
    premise, hypothesis, labels, _, _, _ = preprocess('./dataset/nyt10m_test.txt')

    tokenizer = DebertaTokenizer.from_pretrained('./deberta-large-mnli')
    dataset = NLIDataset(premise, hypothesis, labels, tokenizer, 256)

    test_data_loader = DataLoader(dataset, batch_size=1)
    log.info("-----Dataloader Build!-----")

    if not after_epoch:
        model = Deberta_NLI.from_pretrained('./res/Deberta_NLI/best')
        model.to(device)

    model.eval()
    total_test_loss = 0.
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for batch in test_data_loader:
            sentences = batch['sentences'].to(device)
            labels = batch['labels'].to(device).view(-1)
            attention_mask = batch['attention_masks'].to(device)

            outputs = model(sentences, attention_mask=attention_mask, labels=labels, return_dict=True)
            loss = outputs['loss']
            total_test_loss += loss.item()

            pred = np.argmax(outputs['logits'].detach().cpu().numpy(), axis=1)
            tags = labels.cpu().numpy()

            pred_tags.extend(pred)
            true_tags.extend(tags)

    assert len(pred_tags) == len(true_tags)

    metrics = {}
    metrics['loss'] = total_test_loss / len(test_data_loader)
    metrics['f1'], metrics['precision'], metrics['recall'] = score(true_tags, pred_tags)

    log.info('Test-dataset:\n    precision: {:.2f}%  recall: {:.2f}%  f1: {:.2f}%'.format(metrics['precision'] * 100.,
                                                                                     metrics['recall'] * 100.,
                                                                                     metrics['f1'] * 100.))

    log.info("-----Testing Finished!-----")


def evaluate(model, data_loader):
    model.eval()
    total_loss = 0.
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for batch in data_loader:
            sentences = batch['sentences'].to(device)
            labels = batch['labels'].to(device).view(-1)
            attention_mask = batch['attention_masks'].to(device)

            outputs = model(sentences, attention_mask=attention_mask, labels=labels, return_dict=True)
            loss = outputs['loss']
            total_loss += loss.item()

            pred = np.argmax(outputs['logits'].detach().cpu().numpy(), axis=1)
            tags = labels.cpu().numpy()

            pred_tags.extend(pred)
            true_tags.extend(tags)

    assert len(pred_tags) == len(true_tags)

    metrics = {}
    metrics['loss'] = total_loss / len(data_loader)
    metrics['f1'], metrics['precision'], metrics['recall'] = score(true_tags, pred_tags)

    return metrics


def train(EPOCHS, batch_size, lr, full_fine_tuning=True, resume=False):
    premise_train, hypothesis_train, label_train, _, _, _ = preprocess('./dataset/nyt10m_train.txt')
    premise_val, hypothesis_val, label_val, _, _, _ = preprocess('./dataset/nyt10m_val.txt')

    tokenizer = DebertaTokenizer.from_pretrained('./deberta-large-mnli')
    train_dataset = NLIDataset(premise_train, hypothesis_train, label_train, tokenizer, 256)
    val_dataset = NLIDataset(premise_val, hypothesis_val, label_val, tokenizer, 256)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

    log.info("-----Dataloader Build!-----")

    if not resume:
        model = Deberta_NLI.from_pretrained('./deberta-large-mnli')
    else:
        log.info('-----Resume training from ./res/Deberta_NLI/best!-----')
        model = Deberta_NLI.from_pretrained('./res/Deberta_NLI/best')
    model.to(device)

    if full_fine_tuning:
        deberta_optimizer = list(model.deberta.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in deberta_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in deberta_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': lr * 5, 'weight_decay': 0.01},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': lr * 5, 'weight_decay': 0.}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=len(train_data_loader) * EPOCHS // 10,
                                                num_training_steps=len(train_data_loader) * EPOCHS)

    train_loss = []
    val_loss = []
    val_f1 = []
    max_val_f1 = 0.
    t0 = time.time()
    for epoch in range(EPOCHS):
        total_train_loss = 0.
        model.train()
        for step, batch in enumerate(train_data_loader):
            sentences = batch['sentences'].to(device)
            labels = batch['labels'].to(device).view(-1)
            attention_mask = batch['attention_masks'].to(device)

            model.zero_grad()
            outputs = model(sentences, attention_mask=attention_mask, labels=labels, return_dict=True)
            loss = outputs['loss']
            total_train_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()
            scheduler.step()

            if step % (len(train_data_loader) // 9) == 0:
                log.info("epoch: {} step: {}/{}   {}".format(epoch, step, len(train_data_loader),
                                                               format_time(time.time() - t0)))

        model.save_pretrained('./res/Deberta_NLI/last')

        avg_train_loss = total_train_loss / len(train_data_loader)
        train_loss.append(avg_train_loss)

        # print("Evaluating......")
        train_metrics = evaluate(model, train_data_loader)
        val_metrics = evaluate(model, val_data_loader)

        val_loss.append(val_metrics['loss'])
        val_f1.append(val_metrics['f1'])

        if val_metrics['f1'] > max_val_f1:
            max_val_f1 = val_metrics['f1']
            model.save_pretrained('./res/Deberta_NLI/best')
            log.info("-----Best Model Saved!-----")

        # print("-----------------------------------------------------------------------------")
        log.info("epoch: {}  train_loss: {}  val_loss: {}\n".format(epoch, avg_train_loss, val_metrics['loss']) +
                 "   train: precision: {:.2f}%  recall: {:.2f}%  f1: {:.2f}%\n".format(train_metrics['precision'] * 100.,
                                                                                     train_metrics['recall'] * 100.,
                                                                                     train_metrics['f1'] * 100.) +
                 "   val: precision: {:.2f}%  recall: {:.2f}%  f1: {:.2f}%".format(val_metrics['precision'] * 100.,
                                                                                   val_metrics['recall'] * 100.,
                                                                                   val_metrics['f1'] * 100.))
        # print("-----------------------------------------------------------------------------")

    log.info('-----Training Finished!-----')
