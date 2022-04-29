import sys
import getopt
import joblib

from data_io import *
from prepare_data import *


def get_args():
    optlist, args = getopt.getopt(sys.argv[1:], 'c:r:o:w:')
    opt_var_dict = {'-c': 'data_dir', '-r': 'input_run', '-o': 'output_dir', '-w': 'working_directory'}
    var_dict = dict()

    for o, a in optlist:
        var = opt_var_dict[o]
        var_dict[var] = a

    return var_dict


def predict(lang, vec_file, clf_file, sys_args):
    df = create_df(data_dir=sys_args['data_dir'], lang=lang, df_columns=df_columns)
    vec = joblib.load(os.path.join(sys_args['save_dir'], vec_file))
    x, _ = prepare_xy(df, tweet_feature_method=tweet_feature_method,
                      return_y=False, is_train=False, vec=vec)
    clf = joblib.load(os.path.join(sys_args['save_dir'], clf_file))
    y_pred = clf.predict(x)
    authlist = prepare_authlist(df[df_columns[0]].to_list())
    combine_and_save_output(authlist, y_pred, lang=lang, output_dir=sys_args['output_dir'])

    return


if __name__ == '__main__':
    sys_vars = get_args()
    sys_vars['save_dir'] = 'save'
    df_columns = ['author_id', 'tweet']
    tweet_feature_method = prepare_tweets_using_tfidf
    os.chdir(sys_vars['working_directory'])

    predict(lang='en', vec_file='en_tfidf_vec.pkl', clf_file='en_svc_clf.pkl', sys_args=sys_vars)
    predict(lang='es', vec_file='es_tfidf_vec.pkl', clf_file='es_svc_clf.pkl', sys_args=sys_vars)
