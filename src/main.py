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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression  
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error  
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split  
from cheroot.wsgi import Server
from cheroot.wsgi import PathInfoDispatcher
df_train, df_test = None, None
config_bean = None
db_connection_g = None
feature_list = None
CONFIG_FILE_NAME = 'config.json'
LOG_FILE_NAME = 'log/al.log'
LOG_FILE_ENCODE = 'utf-8'
IMAGE_IMG_TEMPLATE = 'img_{time_stamp}_{idx}.png'.replace('{time_stamp}',
                                                          str(datetime.now().strftime('%Y%m%d_%H%M%S')))
IMAGE_IMG_IDX = 0
skai_app = Flask(__name__)
skai_app.config['SECRET_KEY'] = os.urandom(24)
skai_app.config['SWAGGER'] = {
    'title': 'Cloud computing environment monitoring model',
    'uiversion': 3,
    'version': '1.1',
    'description': 'Cloud computing environment monitoring and early warning',
    'termsOfService': 'https://www.apache.org/licenses/LICENSE-2.0'
}
logger = logging.getLogger('skai_analysis')
logger.setLevel(logging.INFO)
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
            con_rtn = pymysql.connect(
                                      host=self.db_conn_dict.host,
                                      port=self.db_conn_dict.port,
                                      database=self.db_conn_dict.schema,
                                      user=self.db_conn_dict.username,
                                      password=self.db_conn_dict.password
            )
        except pymysql.err.OperationalError:
            if self._logging is not None:
                self._logging.error(traceback.format_exc())
            con_rtn = None
        return con_rtn
    def __str__(self):
        return '''
            [
                db_conn_dict: {db_conn_dict}
            ]
            '''.format(
                db_conn_dict=self.db_conn_dict.__str__()
        )
    def __eq__(self, other):
        """
        equal
        :param other: other object
        :return: is equal
        """
        if isinstance(other,
                      self.__class__):
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
            '''.format(
                       host=self._host,
                       port=self._port,
                       username=self._username,
                       password=self._password,
                       schema=self._schema
        )
    def __eq__(self,
               other):
        """
        equal
        :param other: other object
        :return: is equal
        """
        if isinstance(other,
                      self.__class__):
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
        self._db_connection_1 = None
        self._train_file = None
        self._test_file = None
        self._valid_file = None
        self._support_path = None
        self._is_notebook = False
        self._is_load_data_from_file = True
    @property
    def db_connection_1(self):
        """
        db_connection is a MySQLConnectionInfo object
        """
        if isinstance(self._db_connection_1,
                      MySQLConnectionInfo):
            return self._db_connection_1
        else:
            return None
    @db_connection_1.setter
    def db_connection_1(self,
                        db_connection_1):
        if isinstance(db_connection_1,
                      MySQLConnectionInfo):
            self._db_connection_1 = db_connection_1
    @property
    def train_file(self):
        """
        train_file
        """
        if isinstance(self._train_file,
                      str):
            return self._train_file
        else:
            return None
    @train_file.setter
    def train_file(self,
                   train_file):
        if train_file is None \
                or isinstance(train_file,
                              str):
            self._train_file = train_file
    @property
    def test_file(self):
        """
        test_file
        """
        if isinstance(self._test_file,
                      str):
            return self._test_file
        else:
            return None
    @test_file.setter
    def test_file(self,
                  test_file):
        if test_file is None \
                or isinstance(test_file,
                              str):
            self._test_file = test_file
    @property
    def valid_file(self):
        """
        valid_file
        """
        if isinstance(self._valid_file,
                      str):
            return self._valid_file
        else:
            return None
    @valid_file.setter
    def valid_file(self,
                   valid_file):
        if valid_file is None \
                or isinstance(valid_file,
                              str):
            self._valid_file = valid_file
    @property
    def support_path(self):
        """
        support_path
        """
        if isinstance(self._support_path,
                      str):
            return self._support_path
        else:
            return None
    @support_path.setter
    def support_path(self,
                     support_path):
        if isinstance(support_path,
                      str):
            self._support_path = support_path
    @property
    def is_notebook(self):
        """
        is_notebook
        """
        if isinstance(self._is_notebook,
                      bool):
            return self._is_notebook
        else:
            return False
    @is_notebook.setter
    def is_notebook(self,
                    is_notebook):
        if isinstance(is_notebook,
                      bool):
            self._is_notebook = is_notebook
    @property
    def is_load_data_from_file(self):
        """
        is_load_data_from_file
        """
        if isinstance(self._is_load_data_from_file,
                      bool):
            return self._is_load_data_from_file
        else:
            return False
    @is_load_data_from_file.setter
    def is_load_data_from_file(self,
                               is_load_data_from_file):
        if isinstance(is_load_data_from_file,
                      bool):
            self._is_load_data_from_file = is_load_data_from_file
    def __str__(self):
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
    def __eq__(self,
               other):
        """
        equal
        :param other: other object
        :return: is equal
        """
        if isinstance(other,
                      self.__class__):
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
        self.config_file = os.path.join(config_folder,
                                        CONFIG_FILE_NAME)
        self._json_config = None
        if environment_name is not None \
                and '' != environment_name:
            self._env_config_file = os.path.join(config_folder,
                                                 'config_' + environment_name + '.json')
        else:
            self._env_config_file = None
    def load(self):
        """
        open&read config file
        :return: ConfigBean object
        """
        json_info_str = None
        with open(self.config_file,
                  'r') \
                as _wrk_cfg_file:
            json_info_str = _wrk_cfg_file.read()
        if json_info_str is None:
            raise AssertionError
        _j_dct = json.loads(json_info_str)
        if isinstance(_j_dct,
                      dict):
            p = ConfigBean()
            p.db_connection_1 = self._split_db_connection(_j_dct,
                                                          'db_connection_1')
            p.train_file = _j_dct['train_file']
            p.test_file = _j_dct['test_file']
            p.valid_file = _j_dct['validation_file']
            p.support_path = _j_dct['support_path']
            p.is_notebook = _j_dct['is_notebook']
            p.is_load_data_from_file = _j_dct['is_load_data_from_file']
            self._json_config = p
        self.load_environment()
        return self._json_config
    def load_environment(self):
        """
        overwrite with environment settings
        """
        if self._env_config_file is None:
            return
        json_info_str = None
        with open(self._env_config_file,
                  'r') \
                as _wrk_cfg_file:
            json_info_str = _wrk_cfg_file.read()
        if json_info_str is None:
            raise AssertionError
        _j_dct = json.loads(json_info_str)
        p = ConfigBean()
        if isinstance(_j_dct,
                      dict):
            p.db_connection_1 = self._split_db_connection(_j_dct,
                                                          'db_connection_1')
            p.train_file = _j_dct.get('train_file')
            p.test_file = _j_dct.get('test_file')
            p.valid_file = _j_dct.get('validation_file')
            p.support_path = _j_dct.get('support_path')
            p.is_notebook = _j_dct.get('is_notebook')
            p.is_load_data_from_file = _j_dct.get('is_load_data_from_file')
        self._merge_db_connection(p,
                                  'db_connection_1')
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
        if isinstance(_j_dct, dict) \
                and _j_dct.get(_json_key) is not None:
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
            return
        if not isinstance(cfg_env,
                          ConfigBean):
            return
        db_cfg_main = getattr(self._json_config,
                              _key_name)
        db_cfg_env = getattr(cfg_env,
                             _key_name)
        if db_cfg_env is None:
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
            return
        if not isinstance(db_cfg_env,
                          ConfigBean):
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
            '''.format(
                       config_file=self.config_file,
                       json_config=self._json_config.__str__(),
                       env_config_file=self._env_config_file
        )
    def __eq__(self,
               other):
        """
        equal
        :param other: other object
        :return: is equal
        """
        if isinstance(other,
                      self.__class__):
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
    global \
        config_bean, \
        logger, \
        db_connection_g
    cm = ConfigManager(in_args.configFolder,
                       in_args.environmentName)
    config_bean = cm.load()
    _log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s() - %(levelname)s - %(message)s')
    _file_handler = logging.FileHandler(
                                        os.path.join(config_bean.support_path,
                                                     LOG_FILE_NAME),
                                        mode='w',
                                        encoding=LOG_FILE_ENCODE
    )
    _file_handler.setFormatter(_log_formatter)
    logger.addHandler(_file_handler)
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(_log_formatter)
    logger.addHandler(_console_handler)
    if in_args.environmentName is not None:
        logger.info('Reading environment : ' + in_args.environmentName)
    else:
        logger.info('Reading environment : default')
    if not config_bean.is_load_data_from_file:
        logger.info('Connecting database.')
        db_connection_g = MySQLConnector(config_bean.db_connection_1,
                                         logger).open_db_connection()
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
        if msg is not None:
            logger.warning('Exception occurs while closing %s database connection.',
                           msg)
        else:
            logger.warning('Exception occurs while closing database connection.')
    else:
        if msg is not None:
            logger.info('%s database connection closed.',
                        msg)
        else:
            logger.info('Database connection closed.')
def release_env():
    """
    release resource
    """
    global \
        logger, \
        db_connection_g
    if db_connection_g is None:
        return
    close_db_connection(db_connection_g,
                        logger,
                        msg='default')
def load_data(train_file=None,
              test_file=None,
              valid_file=None,
              sep_flg='\t',
              is_verify_data=True):
    """
    load data
    :param train_file: train file path
    :param test_file: test file path
    :param valid_file: validation file path
    :param sep_flg: delimiter
    :param is_verify_data: display data head when loaded
    :return: data objects as pandas data frame
    """
    _train_cache = None
    _test_cache = None
    _valid_cache = None
    if train_file is not None \
            and train_file != '':
        _train_cache = pd.read_csv(train_file,
                                   sep=sep_flg)
    else:
        raise FileNotFoundError('train file is required!')
    if test_file is not None \
            and test_file != '':
        _test_cache = pd.read_csv(test_file,
                                  sep=sep_flg)
    else:
        raise FileNotFoundError('test file is required!')
    if valid_file is not None \
            and valid_file != '':
        _valid_cache = pd.read_csv(valid_file,
                                   sep=sep_flg)
    if is_verify_data:
        print(_train_cache.head())
        print(_test_cache.head())
        if _valid_cache is not None:
            print(_valid_cache.head())
    return _train_cache, \
           _test_cache, \
           _valid_cache
def view_dataset_info(data_set=None):
    """
    view dataset info
    """
    if data_set is None:
        return
    assert isinstance(data_set,
                      DataFrame)
    print(data_set.info())
def reduce_mem_usage(data_set=None):
    """
    compress memory usage
    """
    if data_set is None:
        return data_set
    assert isinstance(data_set,
                      DataFrame)
    start_mem = data_set.memory_usage().sum() / 1024 ** 2
    numerics = [
        'int16',
        'int32',
        'int64',
        'float16',
        'float32',
        'float64'
    ]
    for col in data_set.columns:
        col_type = data_set[col].dtypes
        if col_type in numerics:
            c_min = data_set[col].min()
            c_max = data_set[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min \
                        and c_max < np.iinfo(np.int8).max:
                    data_set[col] = data_set[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min \
                        and c_max < np.iinfo(np.int16).max:
                    data_set[col] = data_set[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min \
                        and c_max < np.iinfo(np.int32).max:
                    data_set[col] = data_set[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min \
                        and c_max < np.iinfo(np.int64).max:
                    data_set[col] = data_set[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min \
                        and c_max < np.finfo(np.float16).max:
                    data_set[col] = data_set[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min \
                        and c_max < np.finfo(np.float32).max:
                    data_set[col] = data_set[col].astype(np.float32)
                else:
                    data_set[col] = data_set[col].astype(np.float64)
    end_mem = data_set.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return data_set
def get_column_name_list(data_set=None,
                         is_verify_data=True):
    """
    get column name
    """
    if data_set is None:
        return data_set
    assert isinstance(data_set,
                      DataFrame)
    _column = data_set.columns.tolist()
    if is_verify_data:
        print(', '.join(_column))
    return _column
def draw_box_plot(data_set=None,
                  column_list=None,
                  plt_rows=-1,
                  plt_cols=-1):
    """
    draw box plot
    """
    global \
        config_bean, \
        logger, \
        IMAGE_IMG_TEMPLATE, \
        IMAGE_IMG_IDX
    if data_set is None \
            or column_list is None \
            or plt_rows <= 0 \
            or plt_cols <= 0:
        return
    assert isinstance(data_set,
                      DataFrame)
    assert isinstance(column_list,
                      list)
    plt.figure(figsize=(4 * plt_cols, 4 * plt_rows))
    for i in range(plt_rows):
        plt.subplot(
                    plt_rows,
                    plt_cols,
                    i + 1
        )
        sns.boxplot(
            data=data_set[column_list[i]],
            orient='v',
            width=0.5,
            whis=1.5
        )
        plt.ylabel(column_list[i])
    if config_bean.is_notebook:
        plt.show()
    else:
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
def anomaly_detection_by_box_plot(data_set,
                                  line_scope=[-1.0, 1.0]):
    """
    异常值分析
    """
    global \
        config_bean, \
        logger, \
        IMAGE_IMG_TEMPLATE, \
        IMAGE_IMG_IDX
    if data_set is None:
        return
    assert isinstance(data_set,
                      DataFrame)
    plt.figure(figsize=(24, 14))
    plt.boxplot(
        x=data_set.values,
        labels=data_set.columns
    )
    plt.hlines(
        line_scope,
        0,
        40,
        colors='r'
    )
    if config_bean.is_notebook:
        plt.show()
    else:
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
def find_outliers(model,
                  X,
                  y,
                  sigma=3):
    global \
        config_bean, \
        logger, \
        IMAGE_IMG_TEMPLATE, \
        IMAGE_IMG_IDX
    model.fit(
        X,
        y
    )
    y_pred = pd.Series(model.predict(X),
                       index=y.index)
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()
    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index
    print('R2={:.12f}'.format(
        model.score(
            X,
            y
        )
    ))
    logger.info('R2={:.12f}'.format(
        model.score(
            X,
            y
        )
    ))
    print('mse={:.12f}'.format(
        mean_squared_error(
            y,
            y_pred
        )
    ))
    logger.info('mse={:.12f}'.format(
        mean_squared_error(
            y,
            y_pred
        )
    ))
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
    plt.figure(figsize=(15,
                        5))
    ax_131 = plt.subplot(1,
                         3,
                         1)
    plt.plot(
        y,
        y_pred,
        '.'
    )
    plt.plot(
        y.loc[outliers],
        y_pred.loc[outliers],
        'ro'
    )
    plt.legend([
        'Accepted',
        'Outliers'
    ])
    plt.xlabel('y')
    plt.ylabel('y_pred')
    ax_132 = plt.subplot(1,
                         3,
                         2)
    plt.plot(
        y,
        y - y_pred,
        '.'
    )
    plt.plot(
        y.loc[outliers],
        y.loc[outliers] - y_pred.loc[outliers],
        'ro'
    )
    plt.legend([
        'Accepted',
        'Outliers'
    ])
    plt.xlabel('y')
    plt.ylabel('y_pred')
    ax_133 = plt.subplot(1,
                         3,
                         3)
    z.plot.hist(
        bins=50,
        ax=ax_133
    )
    z.loc[outliers].plot.hist(
        color='r',
        bins=50,
        ax=ax_133
    )
    plt.legend([
        'Accepted',
        'Outliers'
    ])
    plt.xlabel('z')
    if config_bean.is_notebook:
        plt.show()
    else:
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
def ana_qq_kde(ds_train,
               ds_test):
    """
    数据分布情况 - 直方图, Q-Q图, KDE图
    """
    global \
        config_bean, \
        logger, \
        IMAGE_IMG_TEMPLATE, \
        IMAGE_IMG_IDX, \
        feature_list
    if ds_train is None:
        return
    if ds_test is None:
        return
    assert isinstance(ds_train,
                      DataFrame)
    assert isinstance(ds_test,
                      DataFrame)
    feature_list = list(ds_train.columns)
    feature_list.remove('target')
    plt_rows = len(feature_list)
    plt_cols = 6
    zoom_to = 1.0
    plt.figure(figsize=(
        4 * zoom_to * plt_cols,
        4 * zoom_to * plt_rows
    ))
    i = 0
    for col in feature_list:
        i += 1
        plt.subplot(
            plt_rows,
            plt_cols,
            i
        )
        sns.histplot(
            ds_train[col],
            kde=True
        )
        sns.histplot(
            ds_test[col],
            kde=True,
            color='r'
        )
        i += 1
        plt.subplot(
            plt_rows,
            plt_cols,
            i
        )
        res = stats.probplot(
            ds_train[col],
            plot=plt
        )
        i += 1
        plt.subplot(
            plt_rows,
            plt_cols,
            i
        )
        ax = sns.kdeplot(
            ds_train[col],
            color='b',
            shade=True
        )
        ax = sns.kdeplot(
            ds_test[col],
            color='r',
            shade=True
        )
        ax.set_xlabel('KDE - ' + col)
        ax.set_ylabel('Frequency')
        ax = ax.legend([
            'train',
            'test'
        ])
    plt.tight_layout()
    if config_bean.is_notebook:
        plt.show()
    else:
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
def linear_regression_diagram(ds_train,
                              ds_test):
    """
    数据分布情况 - 线性回归关系图
    """
    global \
        config_bean, \
        logger, \
        IMAGE_IMG_TEMPLATE, \
        IMAGE_IMG_IDX, \
        feature_list
    if ds_train is None:
        return
    if ds_test is None:
        return
    assert isinstance(ds_train,
                      DataFrame)
    assert isinstance(ds_test,
                      DataFrame)
    feature_list = list(ds_train.columns)
    feature_list.remove('target')
    drop_target = [
        'V5',
        'V9',
        'V11',
        'V14',
        'V17',
        'V21',
        'V22'
    ]
    ds_train.drop(
        drop_target,
        axis=1,
        inplace=True
    )
    ds_test.drop(
        drop_target,
        axis=1,
        inplace=True
    )
    feature_list = list(ds_train.columns)
    feature_list.remove('target')
    fcols = 4
    frows = len(feature_list)
    plt.figure(figsize=(
        5 * fcols,
        4 * frows
    ))
    i = 0
    for col in feature_list:
        i += 1
        ax = plt.subplot(
            frows,
            fcols,
            i
        )
        sns.regplot(
            x=col,
            y='target',
            data=ds_train,
            ax=ax,
            scatter_kws={'marker': '.',
                         's': 3,
                         'alpha': 0.3},
            line_kws={'color': 'k'}
        )
        ax.set_xlabel(col)
        ax.set_ylabel("Target")
        i += 1
        ax = plt.subplot(
            frows,
            fcols,
            i
        )
        sns.histplot(
            ds_train[col].dropna(),
            kde=True
        )
        plt.xlabel(col)
    if config_bean.is_notebook:
        plt.show()
    else:
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
def heatmap(data_set):
    """
    计算相关性系数
    """
    global \
        config_bean, \
        logger, \
        IMAGE_IMG_TEMPLATE, \
        IMAGE_IMG_IDX
    if data_set is None:
        return
    assert isinstance(data_set,
                      DataFrame)
    pd.set_option('display.max_columns',
                  10)
    pd.set_option('display.max_rows',
                  10)
    train_corr = data_set.corr()
    print('current features: {}'.format(len(train_corr.columns)))
    logger.info('current features: {}'.format(len(train_corr.columns)))
    mask = np.zeros_like(train_corr,
                         dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    mask[np.eye(len(train_corr.columns),
                dtype=bool)] = False
    ax = plt.subplots(figsize=(20, 16))
    ax = sns.heatmap(
        train_corr,
        vmax=.8,
        square=True,
        annot=True,
        mask=mask,
        fmt='0.3f'
    )
    if config_bean.is_notebook:
        plt.show()
    else:
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
def heatmap_2(ds_train,
              ds_test,
              threshold=0.1):
    """
    带有过滤条件的 heatmap
    threshold 这个与之设置的越大，剩下的特征就越少
    """
    global \
        config_bean, \
        logger, \
        IMAGE_IMG_TEMPLATE, \
        IMAGE_IMG_IDX, \
        feature_list
    if ds_train is None:
        return
    if ds_test is None:
        return
    assert isinstance(ds_train,
                      DataFrame)
    assert isinstance(ds_test,
                      DataFrame)
    corrmat = ds_train.corr()
    top_corr_features = corrmat.index[abs(corrmat["target"]) > threshold]
    print('current features: {}'.format(len(top_corr_features)))
    logger.info('current features: {}'.format(len(top_corr_features)))
    s_size = len(top_corr_features)
    mask = np.zeros((s_size,
                     s_size),
                    dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    mask[np.eye(s_size,
                dtype=bool)] = False
    plt.subplots(figsize=(20, 16))
    sns.heatmap(
        ds_train[top_corr_features].corr(),
        annot=True,
        cmap="RdYlGn",
        mask=mask,
        fmt='0.3f'
    )
    if config_bean.is_notebook:
        plt.show()
    else:
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
    drop_target = [c for c in corrmat.columns if c not in top_corr_features]
    print('these features will be dropped:{}'.format(drop_target))
    logger.info('these features will be dropped:{}'.format(drop_target))
    ds_train.drop(
        drop_target,
        axis=1,
        inplace=True
    )
    ds_test.drop(
        drop_target,
        axis=1,
        inplace=True
    )
    feature_list = list(ds_train.columns)
    feature_list.remove('target')
    print(', '.join(feature_list))
    logger.info(', '.join(feature_list))
def feature_tag_scatterplot(data_set):
    """
    绘制特征标签与预测标签（target）的散点图
    """
    global \
        config_bean, \
        logger, \
        IMAGE_IMG_TEMPLATE, \
        IMAGE_IMG_IDX, \
        feature_list
    if data_set is None:
        return
    assert isinstance(data_set,
                      DataFrame)
    plt_rows = len(feature_list)
    plt_cols = 4
    plt.figure(figsize=(
        4 * plt_cols,
        4 * plt_rows
    ))
    i = 0
    for f in feature_list:
        i += 1
        plt.subplot(
            plt_rows,
            plt_cols,
            i
        )
        sns.scatterplot(
            x=data_set[f'{f}'],
            y=data_set['target']
        )
    if config_bean.is_notebook:
        plt.show()
    else:
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
def normal_distribution_check(data_set):
    """
    正态分布检验
    """
    global \
        config_bean, \
        logger, \
        IMAGE_IMG_TEMPLATE, \
        IMAGE_IMG_IDX, \
        feature_list
    if data_set is None:
        return
    assert isinstance(data_set,
                      DataFrame)
    for i in feature_list:
        skew = stats.skew(data_set[f'{i}'])
        print(f'the skew value of feature {i} is {skew}')
        logger.info(f'the skew value of feature {i} is {skew}')
    for i in feature_list:
        kurtosis = stats.kurtosis(data_set[f'{i}'])
        print(f'the kurtosis value of feature {i} is {kurtosis}')
        logger.info(f'the kurtosis value of feature {i} is {kurtosis}')
def normalization(ds_train,
                  ds_test):
    """
    归一化
    """
    global \
        logger, \
        feature_list
    if ds_train is None:
        return
    if ds_test is None:
        return
    assert isinstance(ds_train,
                      DataFrame)
    assert isinstance(ds_test,
                      DataFrame)
    train_x = ds_train[feature_list]
    train_x = train_x.apply(
        scale_minmax,
        axis=0
    )
    print('before merge TARGET')
    logger.info('before merge TARGET')
    train_x['target'] = ds_train['target']
    print('merge TARGET')
    logger.info('merge TARGET')
    return train_x,\
           ds_test.apply(scale_minmax,
                         axis=0)
def scale_minmax(col):
    return (col - col.min()) / (col.max() - col.min())
def box_cox(data_set):
    """
    Box-Cox
    """
    global \
        config_bean, \
        logger, \
        IMAGE_IMG_TEMPLATE, \
        IMAGE_IMG_IDX, \
        feature_list
    if data_set is None:
        return data_set
    assert isinstance(data_set,
                      DataFrame)
    train_data_process = data_set[feature_list]
    train_data_process = train_data_process[feature_list].apply(scale_minmax,
                                                                axis=0)
    total_features = len(feature_list)
    feature_list_left = feature_list[0:total_features]
    train_data_process = pd.concat(
        [train_data_process,
         data_set['target']],
        axis=1)
    fcols = 6
    frows = len(feature_list_left)
    plt.figure(figsize=(
        4 * fcols,
        4 * frows
    ))
    i = 0
    for var in feature_list_left:
        dat = train_data_process[[
            var,
            'target'
        ]].dropna()
        i += 1
        plt.subplot(
            frows,
            fcols,
            i
        )
        sns.histplot(
            dat[var],
            kde=True
        )
        plt.title(var + ' Original')
        plt.xlabel('')
        i += 1
        plt.subplot(
            frows,
            fcols,
            i
        )
        _ = stats.probplot(
            dat[var],
            plot=plt
        )
        plt.title('skew=' + '{:.4f}'.format(stats.skew(dat[var])))
        plt.xlabel('')
        plt.ylabel('')
        i += 1
        plt.subplot(
            frows,
            fcols,
            i)
        plt.plot(
            dat[var],
            dat['target'],
            '.',
            alpha=0.5
        )
        plt.title('corr=' + '{:.2f}'.format(np.corrcoef(dat[var],
                                                        dat['target'])[0][1]))
        i += 1
        plt.subplot(
            frows,
            fcols,
            i
        )
        trans_var, lambda_var = stats.boxcox(dat[var].dropna() + 1)
        trans_var = scale_minmax(trans_var)
        sns.histplot(
            trans_var,
            kde=True
        )
        plt.title(var + ' Transformed')
        plt.xlabel('')
        i += 1
        plt.subplot(
            frows,
            fcols,
            i
        )
        _ = stats.probplot(
            trans_var,
            plot=plt
        )
        plt.title('skew=' + '{:.4f}'.format(stats.skew(trans_var)))
        plt.xlabel('')
        plt.ylabel('')
        i += 1
        plt.subplot(
            frows,
            fcols,
            i
        )
        plt.plot(
            trans_var,
            dat['target'],
            '.',
            alpha=0.5
        )
        plt.title('corr=' + '{:.2f}'.format(np.corrcoef(trans_var,
                                                        dat['target'])[0][1]))
    if config_bean.is_notebook:
        plt.show()
    else:
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
def pca(ds_train,
        ds_test):
    global \
        logger, \
        feature_list
    if ds_train is None:
        return
    if ds_test is None:
        return
    assert isinstance(ds_train,
                      DataFrame)
    assert isinstance(ds_test,
                      DataFrame)
    pca = PCA(n_components=16)
    new_train_pca_16 = pca.fit_transform(ds_train.iloc[:, 0:-1])
    new_test_pca_16 = pca.transform(ds_test)
    new_train_pca_16 = pd.DataFrame(new_train_pca_16,
                                    columns=['F-{}'.format(x) for x in range(new_train_pca_16.shape[1])])
    new_test_pca_16 = pd.DataFrame(new_test_pca_16,
                                   columns=['F-{}'.format(x) for x in range(new_test_pca_16.shape[1])])
    new_train_pca_16['target'] = ds_train['target']
    return new_train_pca_16,\
           new_test_pca_16
def train_liner(ds_train,
                train_test_size=0.8):
    """
    多元线性回归
    """
    global logger
    if ds_train is None:
        return
    assert isinstance(ds_train,
                      DataFrame)
    new_train_pca_16 = ds_train.fillna(0)
    train = new_train_pca_16[new_train_pca_16.columns]
    target = new_train_pca_16['target']
    train_data, test_data, train_target, test_target = train_test_split(
        train,
        target,
        test_size=(1 - train_test_size),
        random_state=0
    )
    clf = LinearRegression()
    clf.fit(
        train_data,
        train_target
    )
    score = mean_squared_error(
        test_target,
        clf.predict(test_data)
    )
    print("LinearRegression:{}".format(score))
    logger.info("LinearRegression:{}".format(score))
    kf = KFold(n_splits=5)
    for k, (train_index, test_index) in enumerate(kf.split(train)):
        train_data, test_data, train_target, test_target = train.values[train_index], \
                                                           train.values[test_index], \
                                                           target[train_index], \
                                                           target[test_index]
        clf = SGDRegressor(
            max_iter=1000,
            tol=1e-3
        )
        clf.fit(
            train_data,
            train_target
        )
        score_train = mean_squared_error(
            train_target,
            clf.predict(train_data)
        )
        score_test = mean_squared_error(
            test_target,
            clf.predict(test_data)
        )
        print('{k} fold SGDRegressor train MSE: {score_train:.12f}'.format(
            k=k,
            score_train=score_train)
        )
        logger.info('{k} fold SGDRegressor train MSE: {score_train:.12f}'.format(
            k=k,
            score_train=score_train)
        )
        print('{k} fold SGDRegressor test  MSE: {score_test:.12f}'.format(
            k=k,
            score_test=score_test)
        )
        logger.info('{k} fold SGDRegressor test  MSE: {score_test:.12f}'.format(
            k=k,
            score_test=score_test)
        )
def UDF_remove_anomaly_values(data_set,
                              column_name,
                              threshold=1.0):
    """
    user declare function
    """
    if data_set is None:
        return
    assert isinstance(data_set,
                      DataFrame)
    before_clean = data_set.shape[0]
    data_set = data_set[data_set[column_name] > threshold]
    print('remove data {} rows.'.format(before_clean - data_set.shape[0]))
    logger.info('remove data {} rows.'.format(before_clean - data_set.shape[0]))
    return data_set
def UDF_fill_nan_value(data_set):
    """
    NaN空值补偿
    """
    if data_set is None:
        return data_set
    assert isinstance(data_set,
                      DataFrame)
    feature_list = list(data_set.columns)
    feature_list.remove('target')
    for c in feature_list:
        print('feature: {}, lost rate {}'.format(
            c,
            (data_set.shape[0] - data_set[c].count()) / data_set.shape[0])
        )
        logger.info('feature: {}, lost rate {}'.format(
            c,
            (data_set.shape[0] - data_set[c].count()) / data_set.shape[0])
        )
    return data_set
if __name__ == '__main__':
    """
    command cli:
        python main.py --env demo
    """
    args = _argparse()
    init_env(args)
    assert config_bean is not None \
           and isinstance(config_bean,
                          ConfigBean)
    df_train, df_test, _ = load_data(config_bean.train_file,
                                     config_bean.test_file,
                                     config_bean.valid_file)
    view_dataset_info(df_train)
    view_dataset_info(df_test)
    df_train = reduce_mem_usage(df_train)
    df_test = reduce_mem_usage(df_test)
    column = get_column_name_list(df_train)
    draw_box_plot(
        df_train,
        column,
        plt_rows=len(column) - 1,
        plt_cols=4
    )
    anomaly_detection_by_box_plot(
        df_train,
        line_scope=[-7.5, 7.5]
    )
    df_train = UDF_remove_anomaly_values(
        df_train,
        'V9',
        threshold=-7.5
    )
    df_test = UDF_remove_anomaly_values(
        df_test,
        'V9',
        threshold=-7.5
    )
    df_train = UDF_fill_nan_value(df_train)
    max_loop = 20
    total_drop = 0
    while True:
        X_train = df_train.iloc[:, 0:-1]
        y_train = df_train.iloc[:, -1]
        outliers = find_outliers(
            Ridge(),
            X_train,
            y_train
        )
        outliers = outliers.to_list()
        print('drop: ', outliers)
        logger.info('drop: ' + str(outliers))
        df_train.drop(
            outliers,
            axis=0,
            inplace=True
        )
        if max_loop <= 0 \
                or 0 == len(outliers):
            break
        max_loop -= 1
        total_drop += len(outliers)
        print('loop:', max_loop)
        logger.info('loop:' + str(max_loop))
    print('total dropped {} lines.'.format(total_drop))
    logger.info('total dropped {} lines.'.format(total_drop))
    ana_qq_kde(
        df_train,
        df_test
    )
    linear_regression_diagram(
        df_train,
        df_test
    )
    heatmap(df_train)
    feature_tag_scatterplot(df_train)
    heatmap_2(
        df_train,
        df_test
    )
    normal_distribution_check(df_train)
    df_train, df_test = normalization(
        df_train,
        df_test
    )
    box_cox(df_train)
    pca_train, pca_test = pca(
        df_train,
        df_test
    )
    train_liner(pca_train)

    # start http server
    server = Server((args.bindingAddress,
                     int(args.bindingPort)),
                    PathInfoDispatcher({'/': skai_app}))
    logger.info('Server listening on {}:{}'.format(args.bindingAddress,
                                                   args.bindingPort))
    try:
        server.start()
    except KeyboardInterrupt:
        # release environment
        release_env()
        server.stop()
