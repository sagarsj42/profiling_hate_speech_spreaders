import joblib
from sklearn.svm import SVC

from data_io import *
from prepare_data import *

data_dir = os.path.join('..', 'data', 'pan21-author-profiling-training-2021-03-14')
output_dir = os.path.join('..', 'res')
save_dir = os.path.join('..', 'save')
df_columns = ['author_id', 'tweet', 'label']

en_df = create_df(data_dir, lang='en', df_columns=df_columns)
verify_with_truth(en_df, data_dir, lang='en')
# en_train, en_dev = get_single_split(en_df, data_dir, lang='en')
x_train, y_train, en_vec = prepare_xy(en_df, prepare_tweets_using_tfidf)
# x_dev, y_dev, vec = prepare_xy(en_dev, prepare_tweets_using_tfidf, is_train=False, vec=en_vec)

en_clf = SVC()
en_clf.fit(x_train, y_train)
y_train_pred = en_clf.predict(x_train)

authlist = prepare_authlist(en_df[df_columns[0]].to_list())
combine_and_save_output(authlist, y_train_pred, lang='en', output_dir=output_dir)

joblib.dump(en_vec, os.path.join(save_dir, 'en_tfidf_vec.pkl'))
joblib.dump(en_clf, os.path.join(save_dir, 'en_svc_clf.pkl'))

print(en_clf.score(x_train, y_train))

es_df = create_df(data_dir, lang='es', df_columns=df_columns)
verify_with_truth(es_df, data_dir, lang='es')
# es_train, es_dev = get_single_split(es_df, data_dir, lang='es')
x_train, y_train, es_vec = prepare_xy(es_df, prepare_tweets_using_tfidf)
# x_dev, y_dev, es_vec = prepare_xy(es_dev, prepare_tweets_using_tfidf, is_train=False, vec=es_vec)

es_clf = SVC()
es_clf.fit(x_train, y_train)
y_train_pred = es_clf.predict(x_train)

authlist = prepare_authlist(es_df[df_columns[0]].to_list())
combine_and_save_output(authlist, y_train_pred, lang='es', output_dir=output_dir)

joblib.dump(es_vec, os.path.join(save_dir, 'es_tfidf_vec.pkl'))
joblib.dump(es_clf, os.path.join(save_dir, 'es_svc_clf.pkl'))

print(es_clf.score(x_train, y_train))
