import warnings
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,\
    ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from dataset import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas_ta as ta
from sklearn.svm import SVC
from mlens.ensemble import SuperLearner
warnings.filterwarnings("ignore")


COMISSION = 0
SPLIT = 0.1


def create_models():
    '''
    create models for machine learning
    '''
    return [
        RandomForestClassifier(n_estimators=100),
        BaggingClassifier(n_estimators=100),
        ExtraTreesClassifier(n_estimators=100),
        AdaBoostClassifier(n_estimators=100),
        KNeighborsClassifier(n_neighbors=100),
        LogisticRegression(solver='lbfgs'),
        DecisionTreeClassifier(),
        GaussianNB(),
        SVC(kernel='rbf', C=0.001, probability=True)
    ]


def create_super_learner(X):
    '''
    Create super learner models
    '''
    ensemble = SuperLearner(scorer=metrics.roc_auc_score,
                            folds=2, shuffle=True, sample_size=len(X))
    models = create_models()
    ensemble.add(models)
    ensemble.add_meta(LogisticRegression(solver='lbfgs'))
    return ensemble


def create_features(data):
    '''
    Create features for models machine learning using
    ta-lib library with pandas_ta.
    We create all posible momentum indicators
    '''
    columns_origin = list(data.columns)
    data.ta.strategy("Momentum")
    columns = [i for i in data.columns if i[0].isupper()]
    columns = [i for i in columns if not i.startswith('QQE')]
    columns = [i for i in columns if not i.startswith('HIL')]
    all_columns = [*columns, *columns_origin]
    return data[all_columns], columns


def preprocessing_dataset(data_binance):
    '''
    Create features for models machine learning
    '''

    data = data_binance.copy()
    data, dirnames_lag = create_features(data)
    data.dropna(inplace=True)
    return data, dirnames_lag


def create_target(data):
    data['expected_returns'] = data['close'].pct_change().shift(-1)
    data['direction'] = 0
    data.loc[data['expected_returns'] >
             data['expected_returns'].quantile(q=0.7), 'direction'] = -1
    data.loc[data['expected_returns'] <
             data['expected_returns'].quantile(q=0.3), 'direction'] = 1
    data.dropna(inplace=True)
    return data


def split_data(data):
    '''
    Split data frame on train dataset and test dataset
    '''
    train, test = train_test_split(
        data, shuffle=False, test_size=SPLIT, random_state=0)
    return train, test


def create_ml_model(train, test, dirnames):
    '''
    Input data frame
    '''
    ss = StandardScaler()
    ss.fit(train[dirnames].values)
    ensemble = create_super_learner(
        ss.transform(train[dirnames].values)
    )
    ensemble.fit(
        ss.transform(train[dirnames].values),
        train['direction'],
    )
    test['predict_Logit'] = ensemble.predict(ss.transform(test[dirnames]))
    test['strat_Logit'] = test['predict_Logit'] * test['expected_returns']
    test['strat_Logit'] = np.where(
        test['strat_Logit'] != 0,
        test['strat_Logit'] - COMISSION, test['strat_Logit']
    )
    return test, ensemble, ss


def check_returns_plot(test):
    np.exp(test[['expected_returns', 'strat_Logit']].cumsum()).plot()
    # print(metrics.classification_report(
    #     test['direction'], test['predict_Logit']))


def check_returns(test):
    return np.exp(test[['expected_returns', 'strat_Logit']].sum())


def find_accuracy(ticker, days, ts):
    data_binance = Dataset().get_data(days=days, ticker=ticker, ts=ts)
    data, dirnames_lag = preprocessing_dataset(data_binance)
    data = create_target(data)
    train, test = split_data(data)
    test, ensemble, ss = create_ml_model(
        train=train, test=test, dirnames=dirnames_lag)
    confusion_matrix = metrics.classification_report(
        test['direction'], test['predict_Logit'])

    check_returns_plot(test=test)
    return {
        'confusion_matrix': confusion_matrix,
        'returns': check_returns(test=test)
    }, ensemble, ss


def recomendation_realtime(parameters):
    analyse_data = load_model()
    ticker = parameters.get('ticker')
    data_binance = Dataset().get_data(days=parameters.get('days'),
                                      ticker=ticker,
                                      ts=parameters.get('ts')
                                      )
    data, dirnames_lag = create_features(data_binance)
    data = data.dropna()
    models = analyse_data.get(ticker).get('models')
    ss = analyse_data.get(ticker).get('processing')
    predict = models.predict(ss.transform(data[dirnames_lag].values))
    return {
        'predict': predict[-1],
        'timeindex': data.index[-1]
    }


def save_model(analyse_data):
    with open('models_lr.p', 'wb') as f:
        pickle.dump(analyse_data, f)


def load_model():
    with open('models_lr.p', 'rb') as f:
        analyse_data = pickle.load(f)
    return analyse_data


def analyse_market(tisker_list, parameters):
    market_data = {}
    for i in tisker_list:
        acc, ensemble, ss = find_accuracy(
            ticker=i,
            days=parameters.get('days'),
            ts=parameters.get('ts')
        )
        market_data[i] = {
            'accuracy': acc,
            'models': ensemble,
            'processing': ss
        }
        print(f'{i}: {acc}')
    return market_data


def validation(parameters, days):
    analyse_data = load_model()
    data_binance = Dataset().get_data(days=parameters.get('days'),
                                      ticker=parameters.get('ticker'),
                                      ts=parameters.get('ts')
                                      )
    data = data_binance.copy().reset_index()
    data, dirnames_lag = preprocessing_dataset(data)

    data['signal'] = 0
    data['proba_down'] = 0
    data['proba_up'] = 0

    ticker = parameters.get('ticker')
    models = analyse_data.get(ticker).get('models')
    ss = analyse_data.get(ticker).get('processing')
    data['signal'] = models.predict(ss.transform(data[dirnames_lag]))

    data['expected_returns'] = data['close'].pct_change().shift(-1)
    data['strat_Logit'] = data['expected_returns'] * data['signal']
    data['strat_Logit'] = np.where(
        data['strat_Logit'] != 0,
        data['strat_Logit'] - COMISSION, data['strat_Logit']
    )
    data = data.set_index('time')
    return data.last(f'{days}D')


def analyse_crypto_market(parameters):
    analyse_data = analyse_market(
        tisker_list=[parameters.get('ticker')],
        parameters=parameters
    )
    save_model(analyse_data)
    print('Market analysis completed')


if __name__ == '__main__':
    parameters = {
        'ticker': 'ADAUSDT',
        'days': 30,
        'lag': 200,
        'ts': '5m'
    }
    # Make research of super machine learning model
    analyse_crypto_market(parameters=parameters)

    # Make prediction of crypto market
    data = validation(parameters=parameters, days=5)
    data['strat_Logit'].cumsum().plot()

    # Make prediction of crypto market in realtime
    signal = recomendation_realtime(parameters=parameters)
    print(signal)
