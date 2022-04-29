import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/sagarsj42/transformers'

import sys
import getopt
import time
import datetime
import random

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from data_io import *


def prepare_dataset(tokenizer, index, tweets, labels):
    input_ids = list()
    attention_masks = list()

    for tweet in tweets:
        encoded_dict = tokenizer.encode_plus(
            tweet,
            add_special_tokens = True,
            max_length = 128,
            pad_to_max_length = True,
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids)
    attention_masks = torch.cat(attention_masks)
    index = torch.tensor(index)
    labels = torch.tensor(labels)
    
    dataset = TensorDataset(index, input_ids, attention_masks, labels)
    
    return dataset

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    
    return str(datetime.timedelta(seconds=elapsed_rounded))


try:
    opts, args = getopt.getopt(sys.argv[1:], 'h', ['batch_size=', 'epochs='])
    
    for o, a in opts:
        if o == '--batch_size':
            batch_size = int(a.strip())
        elif o == '--epochs':
            epochs = int(a.strip())
        else:
            assert False, 'Unhandled option!'
except getopt.GetoptError as err:
    print(err)
    print('Using default values')
    
    batch_size = 16
    epochs = 5

data_dir = os.path.join(os.getcwd(), '..', 'data', 'pan21-author-profiling-training-2021-03-14')
save_dir = os.path.join(os.getcwd(), '..', 'save')
save_path = os.path.join(save_dir, 'bert_large_uncased_best_dev_acc.pth')

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    
en_train, en_dev = get_single_split(data_dir, lang='en')
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

en_train_ds = prepare_dataset(tokenizer, en_train.index.values, en_train.tweet.values, en_train.label.values)
en_dev_ds = prepare_dataset(tokenizer, en_dev.index.values, en_dev.tweet.values, en_dev.label.values)

train_dataloader = DataLoader(
    en_train_ds,
    sampler = RandomSampler(en_train_ds),
    batch_size = batch_size
)

dev_dataloader = DataLoader(
    en_dev_ds,
    sampler = SequentialSampler(en_dev_ds),
    batch_size = batch_size
)

model = BertForSequenceClassification.from_pretrained(
    'bert-large-cased',
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False
)

total_steps = len(train_dataloader) * epochs

optimizer = AdamW(
    model.parameters(),
    lr = 2e-5,
    eps = 1e-8
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = total_steps
)

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)

training_stats = list()
total_t0 = time.time()

for epoch in range(epochs):
    print('\n==== Epoch {:} / {:} ===='.format(epoch + 1, epochs))
    print('Training ....')
    
    t0 = time.time()
    total_train_loss = 0.0
    total_train_accuracy = 0.0
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        if step % 100 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('\t\tBatch {:>5,} of {:>5,}. Elapsed {:}'.format(step, len(train_dataloader), elapsed))
            
        b_index, b_input_ids, b_attention_masks, b_labels = tuple(b.to(device) for b in batch)

        model.zero_grad()

        outputs = model(
            b_input_ids,
            token_type_ids = None,
            attention_mask = b_attention_masks,
            labels = b_labels
        )
        loss = outputs['loss']
        logits = outputs['logits']

        loss.backward()

        total_train_loss += loss.item()
        total_train_accuracy += flat_accuracy(logits.detach().clone().cpu().numpy(),
                                              b_labels.detach().clone().to('cpu').numpy())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    training_time = format_time(time.time() - t0)
    
    print('\tAverage training loss: {0:.2f}'.format(avg_train_loss))
    print('\tAverage training accuracy: {0:.2f}'.format(avg_train_accuracy * 100))
    print('\tTraining took: {:}'.format(training_time))
    print('\nRunning validation ....')

    t0 = time.time()
    total_dev_loss = 0.0
    total_dev_accuracy = 0.0
    model.eval()

    for batch in dev_dataloader:
        b_index, b_input_ids, b_attention_masks, b_labels = tuple(b.to(device) for b in batch)

        with torch.no_grad():
            outputs = model(
                b_input_ids,
                token_type_ids = None,
                attention_mask = b_attention_masks,
                labels = b_labels
            )
            loss = outputs['loss']
            logits = outputs['logits']

        total_dev_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        labels = b_labels.to('cpu').numpy()
        total_dev_accuracy += flat_accuracy(logits, labels)
    
    avg_dev_loss = total_dev_loss / len(dev_dataloader)
    avg_dev_accuracy = total_dev_accuracy / len(dev_dataloader)
    dev_time = format_time(time.time() - t0)
    
    if avg_dev_accuracy > best_dev_accuracy:
        best_dev_accuracy = avg_dev_accuracy
        torch.save(model.state_dict(), save_path)
        
    print('\tAverage dev loss: {0:.2f}'.format(avg_dev_loss))
    print('\tAverage dev accuracy: {0:.2f}%'.format(avg_dev_accuracy * 100))
    print('\tValidation took: {:}'.format(dev_time))
    
    training_stats.append(
        {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'dev_loss': avg_dev_loss,
            'train_accuracy': avg_train_accuracy,
            'dev_accuracy': avg_dev_accuracy,
            'train_time': training_time,
            'dev_time': dev_time
        }
    )

print('Training complete!')
print('Total training took {:} (h:mm:ss)'.format(format_time(time.time() - total_t0)))
