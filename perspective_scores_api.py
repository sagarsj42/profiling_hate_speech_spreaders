import time
import pyarrow.parquet as pq
from tqdm.notebook import tqdm

from googleapiclient import discovery
from googleapiclient.errors import HttpError

from data_io import *
from prepare_data import *

API_KEY = 'AIzaSyAKEKvKdrXhcBil0Chq3itSnobj6nBqVSU'
ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])


def get_perspective_client():
    client = discovery.build(
        'commentanalyzer',
        'v1alpha1',
        developerKey=API_KEY,
        discoveryServiceUrl='https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1',
        static_discovery=False
    )

    return client


def prepare_attribute_labels_for_spanish(attribute_list):
    experimental_attributes = ['IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'THREAT']
    not_avail_attributes = ['SEXUALLY_EXPLICIT', 'FLIRTATION']
    new_attributes = list()

    for attr in attribute_list:
        if attr not in not_avail_attributes:
            if attr in experimental_attributes:
                new_attributes.append(attr + '_EXPERIMENTAL')
            else:
                new_attributes.append(attr)

    return new_attributes


def get_perspective_scores_for_comment(client, text, attributes, language):
    requested_attributes = dict()

    if language == 'es':
        attributes = prepare_attribute_labels_for_spanish(attributes)

    for attr in attributes:
        requested_attributes[attr] = dict()
        requested_attributes[attr]['scoreType'] = 'PROBABILITY'
        requested_attributes[attr]['scoreThreshold'] = 0.0

    analyze_request = {
        'comment': {
            'text': text,
            'type': 'PLAIN_TEXT'
        },
        'requestedAttributes': requested_attributes,
        'languages': [language],
        'doNotStore': True
    }

    response = client.comments().analyze(body=analyze_request).execute()
    output_scores = list()

    for attr in attributes:
        output_scores.append(response['attributeScores'][attr]['summaryScore']['value'])
    output_scores = np.array(output_scores)

    return output_scores


def combine_and_get_perspective_scores(client, tweets, steps, attribute_list, lang, halt_time=1):
    n = len(tweets)
    step_size = int(n / steps)
    scores = list()

    for s in range(steps):
        ind = s * step_size
        step_tweets = tweets[ind: ind + step_size]
        combined_tweet = ' '.join(step_tweets)

        single_step_scores = get_perspective_scores_for_comment(client, combined_tweet, attribute_list, lang)
        time.sleep(halt_time)

        scores.append(single_step_scores)

    return scores


def collect_perspective_scores(authors, tweets, attribute_list, lang, steps=20, halt_time=1):
    n = len(authors)

    df_columns = [ordinal(i + 1) for i in range(steps)]
    scores_df = pd.DataFrame(index=authors, columns=df_columns)
    step_scores = list()

    client = get_perspective_client()

    for i in tqdm(range(n)):
        start = i * 200
        try:
            step_scores = combine_and_get_perspective_scores(
                client, tweets[start: start + 200], steps=steps,
                attribute_list=attribute_list, lang=lang, halt_time=halt_time)
        except HttpError as err:
            if str(err).find('Quota exceeded') > -1:
                print('Quota exceeded, halting for 2 min ....')
                time.sleep(120)
                print('Resuming service ....')

                step_scores = combine_and_get_perspective_scores(
                    client, tweets[start: start + 200], steps=steps,
                    attribute_list=attribute_list, lang=lang, halt_time=2 * halt_time)

            elif str(err).find('Comment text was too many bytes.') > -1:
                step_step_scores = combine_and_get_perspective_scores(
                    client, tweets[start: start + 200],
                    steps=2 * steps, attribute_list=attribute_list, lang=lang, halt_time=halt_time)

                step_scores = list()
                for k in range(0, 2 * steps, 2):
                    s1 = step_step_scores[k]
                    s2 = step_step_scores[k+1]
                    s = np.maximum(s1, s2)
                    step_scores.append(s)

        for j in range(steps):
            scores_df.loc[authors[i], df_columns[j]] = step_scores[j]

    return scores_df


def extract_perspective_scores_for_authors(tweet_df, data_dir, lang, steps):
    authors = tweet_df['author_id'].tolist()
    authors = prepare_authlist(authors)
    filename = lang + '_perspective_scores_' + str(steps) + '.parquet'
    score_df = pq.read_table(os.path.join(data_dir, filename)).to_pandas()

    all_scores = list()
    for author in authors:
        auth_scores = score_df.loc[author, ordinal(1)]

        for i in range(1, steps):
            auth_scores = np.concatenate((auth_scores, score_df.loc[author, ordinal(i + 1)]))
        all_scores.append(auth_scores)
    all_scores = np.array(all_scores)

    return all_scores
