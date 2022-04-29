import os
import glob
import xml.etree.ElementTree as et

# import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def create_df(data_dir, lang, df_columns):
    lang_df = pd.DataFrame(columns=df_columns)

    for auth_file in sorted(glob.glob(os.path.join(data_dir, lang + '/*.xml'))):
        root = None
        try:
            root = et.parse(auth_file).getroot()
        except et.ParseError as err:
            if str(err).find('not well-formed (invalid token)') > -1:
                print('ParseError encountered for file ', auth_file, ' -> using latin-1 encoding')
                with open(auth_file, 'r', encoding='latin-1') as f:
                    xml_string = f.read()
                    root = et.fromstring(xml_string)

        author_id = os.path.basename(auth_file)[:-4]
        tweets = root.findall('./documents/*')
        label = root.get('class') if 'label' in df_columns else None

        for tweet in tweets:
            if 'label' in df_columns:
                temp_df = pd.DataFrame(data=[[author_id, tweet.text, label]], columns=df_columns)
            else:
                temp_df = pd.DataFrame(data=[[author_id, tweet.text]], columns=df_columns)
            lang_df = lang_df.append(temp_df, ignore_index=True)

    return lang_df


def verify_with_truth(df, data_dir, lang, usr_len=200):
    truth_path = os.path.join(data_dir, lang, 'truth.txt')

    if os.path.exists(truth_path):
        with open(truth_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                author, label = line.split(':::')
                author_df = df.loc[df['author_id'] == author.strip()]

                assert author_df.shape[0] == usr_len
                assert (author_df['label'] == label.strip()).all()

    return


def split_authorwise(data_dir, lang, train_ratio=0.8, n_splits=1):
    with open(os.path.join(data_dir, lang, 'truth.txt'), 'r') as f:
        lines = f.readlines()
        author_labels = np.array(list(map(
            lambda l: [l.split(':::')[0].strip(), l.split(':::')[1].strip()], lines)))
        authors = author_labels[:, 0]
        labels = author_labels[:, 1]
        splits = list()

        sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_ratio, random_state=0)
        for train_index, dev_index in sss.split(authors, labels):
            splits.append((authors[train_index], authors[dev_index]))

        return splits


def get_train_dev_from_split(df, author_split, usr_len=200):
    train_split = df.loc[df.author_id.isin(author_split[0])]
    dev_split = df.loc[df.author_id.isin(author_split[1])]

    assert (train_split['author_id'].value_counts() == usr_len).all()
    assert (dev_split['author_id'].value_counts() == usr_len).all()
    assert (train_split['label'].value_counts() == len(author_split[0]) * 100).all()
    assert (dev_split['label'].value_counts() == len(author_split[1]) * 100).all()

    return train_split, dev_split


def get_single_split(df, data_dir, lang):
    # df = pq.read_table(os.path.join(data_dir, lang + '_df.parquet')).to_pandas()
    df.label = pd.to_numeric(df.label)
    author_splits = split_authorwise(data_dir, lang)
    train_split, dev_split = get_train_dev_from_split(df, author_splits[0])

    return train_split, dev_split


def create_xml(author_id, lang, label, filedir):
    root = et.Element('author')
    root.set('id', author_id)
    root.set('lang', lang)
    root.set('type', str(int(label)))

    tree = et.ElementTree(root)
    filename = os.path.join(filedir, author_id + '.xml')

    with open(filename, 'wb') as f:
        tree.write(f)

    return tree


def combine_and_save_output(auth, pred, lang, output_dir):
    output_dir = os.path.join(output_dir, lang)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for a, p in zip(auth, pred):
        create_xml(a, lang, p, output_dir)

    return
