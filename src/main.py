# coding=utf-8
# author ielts_go3@163.com
# Cloud computing environment monitoring and early warning system - Prophet
import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
import seaborn as sns
from pandas import DataFrame
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# init global variables
df_train, df_test = None, None

# ConfigManager()
# cm = None
# config_bean -> cm.load()
config_bean = None
# database connection object
db_connection_g = None


# declare static variable
CONFIG_FILE_NAME = 'config.json'
LOG_FILE_NAME = 'log/al.log'
LOG_FILE_ENCODE = 'utf-8'
IMAGE_IMG_TEMPLATE = 'img_{time_stamp}_{idx}.png'.replace('{time_stamp}', str(datetime.now().strftime('%Y%m%d_%H%M%S')))
IMAGE_IMG_IDX = 0

# declare log
logger = logging.getLogger('skai_analysis')
logger.setLevel(logging.INFO)


# TODO allow create a dummy config json


########################################################################################################################
# System Area
########################################################################################################################
def _argparse():
    """
    get input arguments
    :return: argument object
    """
    parser = argparse.ArgumentParser(description="Cloud computing environment monitoring and early warning system.\n"
                                                 "Copyright ielts_go3@163.com")
    parser.add_argument('--bindingAddress',
                        action='store',
                        dest='bindingAddress',
                        default="0.0.0.0",
                        help='Binding IP address')
    parser.add_argument('--bindingPort',
                        action='store',
                        dest='bindingPort',
                        default="7821",
                        help='Binding Port')
    parser.add_argument('--configFolder',
                        action='store',
                        dest='configFolder',
                        default=".",
                        help='All config files stored in this place.')
    parser.add_argument('--env',
                        action='store',
                        dest='environmentName',
                        default=None,
                        help='Environment name')

    return parser.parse_args()


class MySQLConnector:
    """
    MySQLConnector
    """

    def __init__(self,
                 db_conn_dict,
                 logger=None):
        """
        init
        :param db_conn_dict: connection dict
        :param logger: logger
        """
        if isinstance(db_conn_dict,
                      MySQLConnectionInfo):
            self.db_conn_dict = db_conn_dict
        else:
            raise TypeError('')

        self._logging = logger

    def open_db_connection(self):
        """
        Open database connection
        :return database connection object
        """
        try:
            con_rtn = pymysql.connect(host=self.db_conn_dict.host,
                                      port=self.db_conn_dict.port,
                                      database=self.db_conn_dict.schema,
                                      user=self.db_conn_dict.username,
                                      password=self.db_conn_dict.password)
        except pymysql.err.OperationalError:
            # log error
            if self._logging is not None:
                self._logging.error(traceback.format_exc())
            con_rtn = None

        return con_rtn

    def __str__(self):
        return '''
            [
                db_conn_dict: {db_conn_dict}
            ]
            '''.format(db_conn_dict=self.db_conn_dict.__str__())

    def __eq__(self, other):
        """
        equal
        :param other: other object
        :return: is equal
        """
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __hash__(self):
        """
        hash
        :return: hash code
        """
        return hash(self.__str__())


class MySQLConnectionInfo:
    """
    MySQLConnectionInfo
    """

    def __init__(self):
        self._host = None
        self._port = None
        self._username = None
        self._password = None
        self._schema = None

    @property
    def host(self):
        return self._host

    @host.setter
    def host(self,
             host):
        self._host = host

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self,
             port):
        self._port = port

    @property
    def username(self):
        return self._username

    @username.setter
    def username(self,
                 username):
        self._username = username

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self,
                 password):
        self._password = password

    @property
    def schema(self):
        return self._schema

    @schema.setter
    def schema(self,
               schema):
        self._schema = schema

    def __str__(self):
        return '''
            [
                host: {host},
                port: {port},
                username: {username},
                password: {password},
                schema: {schema}
            ]
            '''.format(host=self._host,
                       port=self._port,
                       username=self._username,
                       password=self._password,
                       schema=self._schema)

    def __eq__(self, other):
        """
        equal
        :param other: other object
        :return: is equal
        """
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __hash__(self):
        """
        hash
        :return: hash code
        """
        return hash(self.__str__())


class ConfigBean:
    """
    ConfigBean
    """

    def __init__(self):
        """
        init
        """
        # database connection info
        self._db_connection_1 = None
        # train file path
        self._train_file = None
        # test file path
        self._test_file = None
        # validation file path
        self._valid_file = None
        # support path
        self._support_path = None
        # is run under jupyter notebook
        self._is_notebook = False
        # is load data from file. true - file; false - connect database
        self._is_load_data_from_file = True

    @property
    def db_connection_1(self):
        """
        db_connection is a MySQLConnectionInfo object
        """
        # make sure only MySQLConnectionInfo can be returned
        if isinstance(self._db_connection_1, MySQLConnectionInfo):
            return self._db_connection_1
        else:
            return None

    @db_connection_1.setter
    def db_connection_1(self, db_connection_1):
        # verify input value's type
        if isinstance(db_connection_1, MySQLConnectionInfo):
            self._db_connection_1 = db_connection_1

    @property
    def train_file(self):
        """
        train_file
        """
        # make sure only string can be returned
        if isinstance(self._train_file, str):
            return self._train_file
        else:
            return None

    @train_file.setter
    def train_file(self, train_file):
        # verify input value's type
        if train_file is None or isinstance(train_file, str):
            self._train_file = train_file

    @property
    def test_file(self):
        """
        test_file
        """
        # make sure only string can be returned
        if isinstance(self._test_file, str):
            return self._test_file
        else:
            return None

    @test_file.setter
    def test_file(self, test_file):
        # verify input value's type
        if test_file is None or isinstance(test_file, str):
            self._test_file = test_file

    @property
    def valid_file(self):
        """
        valid_file
        """
        # make sure only string can be returned
        if isinstance(self._valid_file, str):
            return self._valid_file
        else:
            return None

    @valid_file.setter
    def valid_file(self, valid_file):
        # verify input value's type
        if valid_file is None or isinstance(valid_file, str):
            self._valid_file = valid_file

    @property
    def support_path(self):
        """
        support_path
        """
        # make sure only string can be returned
        if isinstance(self._support_path, str):
            return self._support_path
        else:
            return None

    @support_path.setter
    def support_path(self,
                     support_path):
        # verify input value's type
        if isinstance(support_path, str):
            self._support_path = support_path

    @property
    def is_notebook(self):
        """
        is_notebook
        """
        # make sure only boolean can be returned
        if isinstance(self._is_notebook, bool):
            return self._is_notebook
        else:
            return False

    @is_notebook.setter
    def is_notebook(self, is_notebook):
        # verify input value's type
        if isinstance(is_notebook, bool):
            self._is_notebook = is_notebook

    @property
    def is_load_data_from_file(self):
        """
        is_load_data_from_file
        """
        # make sure only boolean can be returned
        if isinstance(self._is_load_data_from_file, bool):
            return self._is_load_data_from_file
        else:
            return False

    @is_load_data_from_file.setter
    def is_load_data_from_file(self, is_load_data_from_file):
        # verify input value's type
        if isinstance(is_load_data_from_file, bool):
            self._is_load_data_from_file = is_load_data_from_file

    def __str__(self):
        # TODO property
        return '''
            ConfigBean:
                db_connection_1: {db_connection_1}
                train_file: {train_file}
                test_file: {test_file}
                valid_file: {valid_file}
                support_path: {support_path}
                is_notebook: {is_notebook}
                is_load_data_from_file: {is_load_data_from_file}
            '''.format(
                       db_connection_1=self._db_connection_1.__str__(),
                       train_file=self._train_file,
                       test_file=self._test_file,
                       valid_file=self._valid_file,
                       support_path=self._support_path,
                       is_notebook=self._is_notebook,
                       is_load_data_from_file=self._is_load_data_from_file,
                       )

    def __eq__(self, other):
        """
        equal
        :param other: other object
        :return: is equal
        """
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __hash__(self):
        """
        hash
        :return: hash code
        """
        return hash(self.__str__())


class ConfigManager:
    """
    ConfigManager
    """

    def __init__(self,
                 config_folder='.',
                 environment_name=None):
        """
        init
        :param config_folder: config file store folder
        :param environment_name: custom environment name
        """
        self.config_file = os.path.join(config_folder, CONFIG_FILE_NAME)

        self._json_config = None
        if environment_name is not None and '' != environment_name:
            self._env_config_file = os.path.join(config_folder, 'config_' + environment_name + '.json')
        else:
            self._env_config_file = None

    def load(self):
        """
        open&read config file
        :return: ConfigBean object
        """
        json_info_str = None
        with open(self.config_file, 'r') as _wrk_cfg_file:
            json_info_str = _wrk_cfg_file.read()

        if json_info_str is None:
            # should not be none
            raise AssertionError

        _j_dct = json.loads(json_info_str)

        if isinstance(_j_dct, dict):
            p = ConfigBean()
            p.db_connection_1 = self._split_db_connection(_j_dct, 'db_connection_1')
            p.train_file = _j_dct['train_file']
            p.test_file = _j_dct['test_file']
            p.valid_file = _j_dct['validation_file']
            p.support_path = _j_dct['support_path']
            p.is_notebook = _j_dct['is_notebook']
            p.is_load_data_from_file = _j_dct['is_load_data_from_file']
            self._json_config = p

        # load environment settings
        self.load_environment()

        return self._json_config

    def load_environment(self):
        """
        overwrite with environment settings
        """
        # verify specific environment config file is exist
        if self._env_config_file is None:
            return

        json_info_str = None
        with open(self._env_config_file, 'r') as _wrk_cfg_file:
            json_info_str = _wrk_cfg_file.read()

        if json_info_str is None:
            # should not be none
            raise AssertionError

        _j_dct = json.loads(json_info_str)

        p = ConfigBean()
        if isinstance(_j_dct, dict):
            p.db_connection_1 = self._split_db_connection(_j_dct, 'db_connection_1')
            p.train_file = _j_dct.get('train_file')
            p.test_file = _j_dct.get('test_file')
            p.valid_file = _j_dct.get('validation_file')
            p.support_path = _j_dct.get('support_path')
            p.is_notebook = _j_dct.get('is_notebook')
            p.is_load_data_from_file = _j_dct.get('is_load_data_from_file')

        # merge
        self._merge_db_connection(p, 'db_connection_1')
        self._merge_others(p)

    @staticmethod
    def _split_db_connection(_j_dct,
                             _json_key):
        """
        load database info
        :param _j_dct: config info as dict object
        :param _json_key: reading keys as list
        :return: database connection info object
        """
        if isinstance(_j_dct, dict) and _j_dct.get(_json_key) is not None:
            mysql__con_info = MySQLConnectionInfo()
            mysql__con_info.host = _j_dct[_json_key].get("host")
            mysql__con_info.port = _j_dct[_json_key].get("port")
            mysql__con_info.username = _j_dct[_json_key].get("username")
            mysql__con_info.password = _j_dct[_json_key].get("password")
            mysql__con_info.schema = _j_dct[_json_key].get("schema")
            return mysql__con_info
        else:
            return None

    def _merge_db_connection(self,
                             cfg_env,
                             _key_name):
        """
        merge database info
        :param cfg_env: ConfigBean instance
        :param _key_name: group name
        :return: None
        """
        if cfg_env is None:
            # do not set up environment name
            return
        if not isinstance(cfg_env, ConfigBean):
            # variable type error
            return

        # get object
        db_cfg_main = getattr(self._json_config, _key_name)
        db_cfg_env = getattr(cfg_env, _key_name)

        if db_cfg_env is None:
            # override config can not be none
            return

        if db_cfg_env.host is not None:
            db_cfg_main.host = db_cfg_env.host

        if db_cfg_env.port is not None:
            db_cfg_main.port = db_cfg_env.port

        if db_cfg_env.username is not None:
            db_cfg_main.username = db_cfg_env.username

        if db_cfg_env.password is not None:
            db_cfg_main.password = db_cfg_env.password

        if db_cfg_env.schema is not None:
            db_cfg_main.schema = db_cfg_env.schema

    def _merge_others(self,
                      db_cfg_env):
        """
        merge others info
        :param db_cfg_env: ConfigBean instance
        """
        if db_cfg_env is None:
            # do not setup environment name
            return
        if not isinstance(db_cfg_env, ConfigBean):
            # variable type error
            return

        if db_cfg_env.train_file is not None:
            self._json_config.train_file = db_cfg_env.train_file

        if db_cfg_env.test_file is not None:
            self._json_config.test_file = db_cfg_env.test_file

        if db_cfg_env.valid_file is not None:
            self._json_config.valid_file = db_cfg_env.valid_file

        if db_cfg_env.support_path is not None:
            self._json_config.support_path = db_cfg_env.support_path

        if db_cfg_env.is_notebook is not None:
            self._json_config.is_notebook = db_cfg_env.is_notebook

        if db_cfg_env.is_load_data_from_file is not None:
            self._json_config.is_load_data_from_file = db_cfg_env.is_load_data_from_file

    def __str__(self):
        return '''
            [
                config_file: {config_file},
                json_config: {json_config},
                env_config_file: {env_config_file}
            ]
            '''.format(config_file=self.config_file,
                       json_config=self._json_config.__str__(),
                       env_config_file=self._env_config_file)

    def __eq__(self, other):
        """
        equal
        :param other: other object
        :return: is equal
        """
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __hash__(self):
        """
        hash
        :return: hash code
        """
        return hash(self.__str__())


def init_env(in_args):
    """
    init environment variables
    :param in_args: input argument
    """

    global config_bean, logger, db_connection_g

    # config =====
    cm = ConfigManager(in_args.configFolder,
                       in_args.environmentName)
    config_bean = cm.load()

    # log file =====
    _log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s() - %(levelname)s - %(message)s')
    _file_handler = logging.FileHandler(os.path.join(config_bean.support_path, LOG_FILE_NAME),
                                        mode='w',
                                        encoding=LOG_FILE_ENCODE)
    _file_handler.setFormatter(_log_formatter)
    logger.addHandler(_file_handler)

    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(_log_formatter)
    logger.addHandler(_console_handler)

    # environment =====
    if in_args.environmentName is not None:
        logger.info('Reading environment : ' + in_args.environmentName)
    else:
        logger.info('Reading environment : default')

    if not config_bean.is_load_data_from_file:
        # load from database
        logger.info('Connecting database.')

        db_connection_g = MySQLConnector(config_bean.db_connection_1, logger).open_db_connection()
        if db_connection_g is None:
            logger.error('Database connection error.')
            sys.exit()

        logger.info('Database connected.')


def close_db_connection(db_c,
                        logger,
                        msg=None):
    """
    close database connection and ignore any error
    :param db_c: database connection object
    :param logger: logger
    :param msg: message
    """
    try:
        db_c.close()
    except Exception:
        # ignore error
        if msg is not None:
            logger.warning('Exception occurs while closing %s database connection.', msg)
        else:
            logger.warning('Exception occurs while closing database connection.')
    else:
        if msg is not None:
            logger.info('%s database connection closed.', msg)
        else:
            logger.info('Database connection closed.')


def release_env():
    """
    release resource
    """
    global logger, db_connection_g

    # verify
    if db_connection_g is None:
        return

    # close database
    close_db_connection(db_connection_g,
                        logger,
                        msg='default')


########################################################################################################################
# Logic Area
########################################################################################################################


def load_data(train_file=None, test_file=None, valid_file=None, sep_flg='\t', is_verify_data=True):
    """
    load data
    :param train_file: train file path
    :param test_file: test file path
    :param valid_file: validation file path
    :param sep_flg: delimiter
    :param is_verify_data: display data head when loaded
    :return: data objects as pandas data frame
    """

    # init
    _train_cache = None
    _test_cache = None
    _valid_cache = None

    # load data
    if train_file is not None and train_file != '':
        _train_cache = pd.read_csv(train_file, sep=sep_flg)
    else:
        raise FileNotFoundError('train file is required!')

    if test_file is not None and test_file != '':
        _test_cache = pd.read_csv(test_file, sep=sep_flg)
    else:
        raise FileNotFoundError('test file is required!')

    if valid_file is not None and valid_file != '':
        _valid_cache = pd.read_csv(valid_file, sep=sep_flg)

    # verify data
    if is_verify_data:
        # 训练集
        print(_train_cache.head())
        # 测试集
        print(_test_cache.head())
        # 验证集
        if _valid_cache is not None:
            print(_valid_cache.head())

    return _train_cache, _test_cache, _valid_cache


def view_dataset_info(data_set=None):
    """
    view dataset info
    """
    # verify
    if data_set is None:
        return

    assert isinstance(data_set, DataFrame)
    print(data_set.info())


def reduce_mem_usage(data_set=None):
    """
    compress memory usage
    """
    # verify
    if data_set is None:
        return data_set

    assert isinstance(data_set, DataFrame)

    # calculate original memory size
    start_mem = data_set.memory_usage().sum() / 1024 ** 2
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in data_set.columns:
        col_type = data_set[col].dtypes
        if col_type in numerics:
            c_min = data_set[col].min()
            c_max = data_set[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data_set[col] = data_set[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data_set[col] = data_set[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data_set[col] = data_set[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data_set[col] = data_set[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data_set[col] = data_set[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data_set[col] = data_set[col].astype(np.float32)
                else:
                    data_set[col] = data_set[col].astype(np.float64)

    end_mem = data_set.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return data_set


def get_column_name_list(data_set=None, is_verify_data=True):
    """
    get column name
    """
    # verify
    if data_set is None:
        return data_set

    assert isinstance(data_set, DataFrame)

    # 获取数据的列名称
    _column = data_set.columns.tolist()
    if is_verify_data:
        print(', '.join(_column))

    return _column


def draw_box_plot(data_set=None, column_list=None, plt_rows=-1, plt_cols=-1):
    """
    draw box plot
    """

    global config_bean, logger, IMAGE_IMG_TEMPLATE, IMAGE_IMG_IDX

    # verify
    if data_set is None or column_list is None or plt_rows <= 0 or plt_cols <= 0:
        return

    assert isinstance(data_set, DataFrame)
    assert isinstance(column_list, list)

    # declare image size
    plt.figure(figsize=(4 * plt_cols, 4 * plt_rows))

    for i in range(plt_rows):
        plt.subplot(plt_rows, plt_cols, i + 1)
        sns.boxplot(data=data_set[column_list[i]], orient='v', width=0.5, whis=1.5)
        plt.ylabel(column_list[i])

    if config_bean.is_notebook:
        plt.show()
    else:
        # generate output file name
        _out_img = os.path.join(
            config_bean.support_path,
            IMAGE_IMG_TEMPLATE.format(
                idx=IMAGE_IMG_IDX
            )
        )
        logger.info('save image to {0}'.format(_out_img))

        plt.savefig(_out_img)

        plt.close()
        IMAGE_IMG_IDX += 1


def anomaly_detection_by_box_plot(data_set, line_scope=[-1.0, 1.0]):
    """
    异常值分析
    """

    global config_bean, logger, IMAGE_IMG_TEMPLATE, IMAGE_IMG_IDX

    # verify
    if data_set is None:
        return

    assert isinstance(data_set, DataFrame)

    plt.figure(figsize=(24, 14))
    plt.boxplot(x=data_set.values, labels=data_set.columns)

    plt.hlines(line_scope, 0, 40, colors='r')

    if config_bean.is_notebook:
        plt.show()
    else:
        # generate output file name
        _out_img = os.path.join(
            config_bean.support_path,
            IMAGE_IMG_TEMPLATE.format(
                idx=IMAGE_IMG_IDX
            )
        )
        logger.info('save image to {0}'.format(_out_img))

        plt.savefig(_out_img)

        plt.close()
        IMAGE_IMG_IDX += 1


# function to detect outliers based on the predictions of a model
def find_outliers(model, X, y, sigma=3):  # X:feature y:label  return: 异常值的index
    # predict y values using model
    # try:
    #     y_pred = pd.Series(model.predict(X), index=y.index)
    # # if predicting failed, try fitting the model first
    # except:
    #     model.fit(X, y)
    #     y_pred = pd.Series(model.predict(X), index=y.index)

    global config_bean, logger, IMAGE_IMG_TEMPLATE, IMAGE_IMG_IDX

    model.fit(X, y)
    y_pred = pd.Series(model.predict(X), index=y.index)

    # calculate residuals between the model prediction and true y values
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # calculate a statistic, define outliers to be where z >sigma 这里是标准化的计算公式
    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index

    # print and plot the results
    print('R2={:.12f}'.format(model.score(X, y)))
    logger.info('R2={:.12f}'.format(model.score(X, y)))

    print('mse={:.12f}'.format(mean_squared_error(y, y_pred)))
    logger.info('mse={:.12f}'.format(mean_squared_error(y, y_pred)))

    print('-----------------------------------')
    logger.info('-----------------------------------')

    print('mean of residuals:{:.12f}'.format(mean_resid))
    logger.info('mean of residuals:{:.12f}'.format(mean_resid))

    print('std of residuals:{:.12f}'.format(std_resid))
    logger.info('std of residuals:{:.12f}'.format(std_resid))

    print('-----------------------------------')
    logger.info('-----------------------------------')

    print('{} outliers:'.format(len(outliers)))
    logger.info('{} outliers:'.format(len(outliers)))

    print(outliers.tolist())
    logger.info(outliers.tolist())

    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, '.')
    plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outliers'])
    plt.xlabel('y')
    plt.ylabel('y_pred')

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.')
    plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outliers'])
    plt.xlabel('y')
    plt.ylabel('y_pred')

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    plt.legend(['Accepted', 'Outliers'])
    plt.xlabel('z')

    if config_bean.is_notebook:
        plt.show()
    else:
        # generate output file name
        _out_img = os.path.join(
            config_bean.support_path,
            IMAGE_IMG_TEMPLATE.format(
                idx=IMAGE_IMG_IDX
            )
        )
        logger.info('save image to {0}'.format(_out_img))

        plt.savefig(_out_img)

        plt.close()
        IMAGE_IMG_IDX += 1

    return outliers


def ana_qq_kde(ds_train, ds_test):
    """
    数据分布情况 - 直方图, Q-Q图, KDE图
    """

    global config_bean, logger, IMAGE_IMG_TEMPLATE, IMAGE_IMG_IDX

    # verify
    if ds_train is None:
        return
    if ds_test is None:
        return

    assert isinstance(ds_train, DataFrame)
    assert isinstance(ds_test, DataFrame)

    # 提取df_train中的特征标签，并将起转换成列表形式
    feature_list = list(ds_train.columns)
    # 为方便之后使用，去掉列表中被一并提取出来的target标签，确保仅留特征标签
    feature_list.remove('target')

    plt_rows = len(feature_list)
    plt_cols = 6
    zoom_to = 1.0

    plt.figure(figsize=(4 * zoom_to * plt_cols, 4 * zoom_to * plt_rows))

    i = 0
    for col in feature_list:
        i += 1
        plt.subplot(plt_rows, plt_cols, i)
        sns.histplot(ds_train[col], kde=True)
        sns.histplot(ds_test[col], kde=True, color='r')

        i += 1
        plt.subplot(plt_rows, plt_cols, i)
        res = stats.probplot(ds_train[col], plot=plt)

        i += 1
        plt.subplot(plt_rows, plt_cols, i)
        ax = sns.kdeplot(ds_train[col], color='b', shade=True)
        ax = sns.kdeplot(ds_test[col], color='r', shade=True)
        ax.set_xlabel('KDE - ' + col)
        ax.set_ylabel('Frequency')
        ax = ax.legend(['train', 'test'])

    plt.tight_layout()
    if config_bean.is_notebook:
        plt.show()
    else:
        # generate output file name
        _out_img = os.path.join(
            config_bean.support_path,
            IMAGE_IMG_TEMPLATE.format(
                idx=IMAGE_IMG_IDX
            )
        )
        logger.info('save image to {0}'.format(_out_img))

        plt.savefig(_out_img)

        plt.close()
        IMAGE_IMG_IDX += 1


def UDF_remove_anomaly_values(data_set, column_name, threshold=1.0):
    """
    user declare function
    """

    # verify
    if data_set is None:
        return

    assert isinstance(data_set, DataFrame)

    # backup row amounts
    before_clean = data_set.shape[0]

    data_set = data_set[data_set[column_name] > threshold]

    # display(data_set.describe())

    print('remove data {} rows.'.format(before_clean - data_set.shape[0]))
    logger.info('remove data {} rows.'.format(before_clean - data_set.shape[0]))

    return data_set


def UDF_fill_nan_value(data_set):
    """
    NaN空值补偿
    """

    # verify
    if data_set is None:
        return data_set

    assert isinstance(data_set, DataFrame)

    # 检查每一个特征的缺失情况

    # 提取df_train中的特征标签，并将起转换成列表形式
    feature_list = list(data_set.columns)
    # 为方便之后使用，去掉列表中被一并提取出来的target标签，确保仅留特征标签
    feature_list.remove('target')

    for c in feature_list:
        print('feature: {}, lost rate {}'.format(c, (data_set.shape[0] - data_set[c].count()) / data_set.shape[0]))
        logger.info('feature: {}, lost rate {}'.format(c, (data_set.shape[0] - data_set[c].count()) / data_set.shape[0]))

    return data_set


if __name__ == '__main__':
    """
    command cli:
        python main.py --env demo
    """

    # global config_bean, logger

    # get arguments
    args = _argparse()
    # load config
    init_env(args)

    # valid
    assert config_bean is not None and isinstance(config_bean, ConfigBean)

    # start http server
    # server = Server((args.bindingAddress,
    #                  int(args.bindingPort)),
    #                 PathInfoDispatcher({'/': skai_app}))
    # logger.info('Server listening on {}:{}'.format(args.bindingAddress,
    #                                                args.bindingPort))
    # try:
    #     server.start()
    # except KeyboardInterrupt:
    #     # release environment
    #     release_env()
    #     server.stop()

    df_train, df_test, _ = load_data(config_bean.train_file, config_bean.test_file, config_bean.valid_file)

    view_dataset_info(df_train)
    view_dataset_info(df_test)
    df_train = reduce_mem_usage(df_train)
    df_test = reduce_mem_usage(df_test)

    # get column names
    column = get_column_name_list(df_train)
    # draw box plot, do not contain 'target' column
    draw_box_plot(df_train, column, plt_rows=len(column) - 1, plt_cols=4)

    # 异常值分析
    # TODO 这里为什么是在[-7.5, 7.5]划分出一个区间？
    anomaly_detection_by_box_plot(df_train, line_scope=[-7.5, 7.5])

    df_train = UDF_remove_anomaly_values(df_train, 'V9', threshold=-7.5)
    df_test = UDF_remove_anomaly_values(df_test, 'V9', threshold=-7.5)

    df_train = UDF_fill_nan_value(df_train)

    # 数据分布情况 - 岭回归获取异常值
    # 以循环方式按行删除所有异常数据， 最多 20 次循环
    max_loop = 20
    total_drop = 0
    while True:

        X_train = df_train.iloc[:, 0:-1]
        y_train = df_train.iloc[:, -1]
        outliers = find_outliers(Ridge(), X_train, y_train)
        outliers = outliers.to_list()
        print('drop: ', outliers)
        logger.info('drop: ' + str(outliers))
        df_train.drop(outliers, axis=0, inplace=True)

        if max_loop <= 0 or 0 == len(outliers):
            break

        max_loop -= 1
        total_drop += len(outliers)
        print('loop:', max_loop)
        logger.info('loop:' + str(max_loop))

    print('total dropped {} lines.'.format(total_drop))
    logger.info('total dropped {} lines.'.format(total_drop))

    # 数据分布情况 - 直方图, Q-Q图, KDE图
    ana_qq_kde(df_train, df_test)
