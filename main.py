from typing import Tuple, Dict, List
from pandas import DataFrame 
import time
from nilmtk import DataSet, MeterGroup
import loguru
from numba import njit
import numpy as np
import pandas as pd
import sys
from fuzzywuzzy import fuzz
from enum import Enum
from skmultilearn.adapt import MLkNN
from skmultilearn.ensemble import RakelD
from sklearn.neural_network import MLPClassifier
from pyts import approximation, transformation
from sklearn.metrics import f1_score

SITE_METER = 'Site meter'

'''

'''
DEBUG: bool = True
TIMING: bool = True
TRACE_MEMORY: bool = True
INFO: bool = True
MB: int = 1024 * 1024
mlknn = MLkNN(ignore_first_neighbours=0, k=3, s=1.0)
rakel = RakelD(MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive',solver='adam'), labelset_size=5)

def read_REDD(datasource, start, end, sample_period = 6, building = 1) -> Tuple[DataFrame, MeterGroup]:
    datasource.set_window(start = start, end = end)
    redd_meter = datasource.buildings[building].elec.mains()
    
    if isinstance(redd_meter, MeterGroup):
        mains_metergroup = redd_meter
    else:
        mains_metergroup = MeterGroup(meters = [redd_meter])
    start_time = time.time() if TIMING else None
    df = mains_metergroup.dataframe_of_meters(sample_period=sample_period)
    df.fillna(0, inplace=True)
    return df, mains_metergroup

'''

DEBUGGING + TIMING

'''
class NoSiteMeterException(Exception):
    pass


class LabelNormalizationError(Exception):
    pass


def debug(d):
    if DEBUG:
        print('DEBUG: ' + d)


def info(i):
    if INFO:
        print('INFO: ' + i)


def timing(t):
    if TIMING:
        print('TIMING: ' + t)


def debug_mem(message, obj):
    if TRACE_MEMORY:
        print('MEMORY: {}'.format(message.format(sys.getsizeof(obj) / MB)))


def trace_mem(o):
    return sys.getsizeof(o) / MB


def array_info(ar):
    print(f'type :{type(ar)}; dtype:{ar.dtype}; ndim={ar.ndim}; shape:{ar.shape}')

"""
CHAOTIC_TOOLKIT
"""

def takens_embedding(series: np.ndarray, delay, dimension) -> np.ndarray:
    """
    This function returns the Takens embedding of data with delay into dimension,
    delay*dimension must be < len(data)
    """
    if delay * dimension > len(series):
        info(f'Not enough data for the given delay ({delay}) and dimension ({dimension}).'
             f'\ndelay * dimension > len(data): {delay * dimension} > {len(series)}')
        return series
    delay_embedding = np.array([series[0:len(series) - delay * dimension]])
    for i in range(1, dimension):
        delay_embedding = np.append(delay_embedding,
                                    [series[i * delay:len(series) - delay * (dimension - i)]], axis=0)
    return delay_embedding
"""
TIME SERIES LENGTH

"""
class TimeSeriesLength(Enum):
    WINDOW_SAMPLE_PERIOD = 'same'
    WINDOW_1_MIN = '1m'
    WINDOW_5_MINS = '5m'
    WINDOW_10_MINS = '10m'
    WINDOW_30_MINS = '30m'
    WINDOW_1_HOUR = '1h'
    WINDOW_2_HOURS = '2h'
    WINDOW_4_HOURS = '4h'
    WINDOW_8_HOURS = '8h'
    WINDOW_1_DAY = '1d'
    WINDOW_1_WEEK = '1w'

def read_all_meters(dataset, start: str, end: str, sample_period: int = 6, building: int = 1) \
            -> Tuple[DataFrame, MeterGroup]:
        """
        Read the records during the given start and end dates, for all the meters of the given building.
        Args:
            start (str): The starting date in the format "{month}-{day of month}-{year}" e.g. "05-30-2012".
            end (str): The final date in the format "{month}-{day of month}-{year}" e.g. "08-30-2012".
            sample_period (int): The sample period of the records.
            building (int): The building to read the records from.
        Returns:
            Returns a tuple containing the respective DataFrame and MeterGroup of the data that are read.
        """
        start_time = time.time() if TIMING else None
        dataset.set_window(start=start, end=end)
        elec = dataset.buildings[building].elec
        timing('NILMTK selecting all meters: {}'.format(round(time.time() - start_time, 2)))

        start_time = time.time() if TIMING else None
        df = elec.dataframe_of_meters(sample_period=sample_period)
        timing('NILMTK converting all meters to dataframe: {}'.format(round(time.time() - start_time, 2)))

        df.fillna(0, inplace=True)
        
        return df, elec

    
def read_selected_appliances( appliances: List, start: str, end: str, sample_period=6, building=1,
                                 include_mains=True) -> Tuple[DataFrame, MeterGroup]:
        """
        Loads the data of the specified appliances.
        Args:
            appliances (List): A list of appliances to read their records.
            start (str): The starting date in the format "{month}-{day of month}-{year}" e.g. "05-30-2012".
            end (str): The final date in the format "{month}-{day of month}-{year}" e.g. "08-30-2012".
            sample_period (int): The sample period of the records.
            building (int): The building to read the records from.
            include_mains (bool): True if should include main meters.
        Returns:
            Returns a tuple containing the respective DataFrame and MeterGroup of the data that are read.
        """
        debug(f" read_selected_appliances {appliances}, {building}, {start}, {end}, {include_mains}")

        selected_metergroup = get_selected_metergroup(redd, appliances, building, end, start, include_mains)

        start_time = time.time() if TIMING else None
        df = selected_metergroup.dataframe_of_meters(sample_period=sample_period)
        timing('NILMTK converting specified appliances to dataframe: {}'.format(round(time.time() - start_time, 2)))

        debug(f"Length of data of read_selected_appliances {len(df)}")
        df.fillna(0, inplace=True)
        print('\n')
        print('READ_SELECTED_APPLIANCES: ')
        print(df.head())
        
        return df, selected_metergroup

def get_selected_metergroup(dataset, appliances, building, end, start, include_mains) -> MeterGroup:
        """
        Gets a MeterGroup with the specified appliances for the given building during the given dates.
        Args:
            appliances (List): A list of appliances to read their records.
            building (int): The building to read the records from.
            start (str): The starting date in the format "{month}-{day of month}-{year}" e.g. "05-30-2012".
            end (str): The final date in the format "{month}-{day of month}-{year}" e.g. "08-30-2012".
            include_mains (bool): True if should include main meters.
        Returns:
            A MeterGroup containing the specified appliances.
        """
        start_time = time.time() if TIMING else None
        dataset.set_window(start=start, end=end)
        elec = dataset.buildings[building].elec
        appliances_with_one_meter = []
        appliances_with_more_meters = []
        for appliance in appliances:
            metergroup = elec.select_using_appliances(type=appliances)
            if len(metergroup.meters) > 1:
                appliances_with_more_meters.append(appliance)
            else:
                appliances_with_one_meter.append(appliance)

        special_metergroup = None
        for appliance in appliances_with_more_meters:
            inst = 1
            if appliance == 'sockets' and building == 3:
                inst = 4
            if special_metergroup is None:
                special_metergroup = elec.select_using_appliances(type=appliance, instance=inst)
            else:
                special_metergroup = special_metergroup.union(elec.select_using_appliances(type=appliance, instance=1))

        selected_metergroup = elec.select_using_appliances(type=appliances_with_one_meter)
        selected_metergroup = selected_metergroup.union(special_metergroup)
        if include_mains:
            mains_meter = dataset.buildings[building].elec.mains()
            if isinstance(mains_meter, MeterGroup):
                if len(mains_meter.meters) > 1:
                    mains_meter = mains_meter.meters[0]
                    mains_metergroup = MeterGroup(meters=[mains_meter])
                else:
                    mains_metergroup = mains_meter
            else:
                mains_metergroup = MeterGroup(meters=[mains_meter])
            selected_metergroup = selected_metergroup.union(mains_metergroup)
        timing('NILMTK select using appliances: {}'.format(round(time.time() - start_time, 2)))
        return selected_metergroup
    

def normalize_columns(df: DataFrame, meter_group: MeterGroup, appliance_names: List[str]) ->Tuple[DataFrame, dict]: 
    labels = meter_group.get_labels(df.columns)
    normalized_labels = []
    info(f"Df columns before normalization {df.columns}")
    info(f"Labels before normalization {labels}")

    for label in labels:
        if label == SITE_METER and SITE_METER not in appliance_names:
            normalized_labels.append(SITE_METER)
            continue
        for name in appliance_names:
            ratio = fuzz.ratio(label.lower().replace('electric', "").lstrip().rstrip().split()[0],
                               name.lower().replace('electric', "").lstrip().rstrip().split()[0])
            if ratio > 90:
                info(f"{name} ~ {label} ({ratio}%)")
                normalized_labels.append(name)
                '''
    if len(normalized_labels) != len(labels):
        debug(f"len(normalized_labels) {len(normalized_labels)} != len(labels) {len(labels)}")
        raise LabelNormalizationError()
        '''
    label2id = {l: i for l, i in zip(normalized_labels, df.columns)}
    df.columns = normalized_labels
    info(f"Normalized labels {normalized_labels}")
    return df, label2id
'''
CREATE MULTILABELS FROM METERS
'''

def create_multilabels_from_meters(meters: DataFrame, meter_group: MeterGroup, labels2id: dict) -> DataFrame:
    start_time = time.time() if TIMING else None
    labels = dict()
    for col in meters.columns:
        loguru.logger.info(f"Creating multilabels from meter {col}, "
                           f"\nlabels2id[col] {labels2id[col]}"
                           f"\nmetergroup[labels2id[col]] {meter_group[labels2id[col]]}")
        meter = meter_group[labels2id[col]]
        threshold = meter.on_power_threshold()
        print("threshold = ", threshold)
        vals = meters[col].values.astype(float)
        if vals is None or col == SITE_METER:
            loguru.logger.debug(f"Skipping {col} - {vals}")
            continue
        loguru.logger.debug(f"meters[col].values.astype(float) {col} - {vals}")
        labels[col] = create_labels(vals, threshold)
    timing('Create multilabels from meters {}'.format(round(time.time() - start_time, 2)))
    return DataFrame(labels)



@njit(parallel=True)
def create_labels(array, threshold):
    res = np.empty(array.shape)
    for i in range(len(array)):
        if array[i] >= threshold:
            res[i] = 1
        else:
            res[i] = 0
    return list(res)



'''
SET UP 1 BUILDING 
'''
def setup_one_building(appliances, datasource, building, start_date, end_date,
                           sample_period) -> (pd.DataFrame, MeterGroup, Dict, Dict):
        """
        Setup and load the data using one building.
        Args:
            appliances (List): The appliances that will be recongized.
            datasource (Datasource): The Datasource that will be used to load energy data.
            building (int): The building that is used.
            start_date (str): Start date of the data that will be selected for each building.
            end_date (str): End date of the data that will be selected for each building.
            sample_period (int): The sampling frequency.
        Returns:
        """
        if appliances:
            info(f'Reading data from specified meters. \n-Building: {building}\n-Appliances {appliances}')
            all_df, metergroup = read_selected_appliances(appliances=appliances, start=start_date,
                                                                     end=end_date,
                                                                     sample_period=sample_period, building=building)

        else:
            info('Reading data from all meters...')
            all_df, metergroup = read_all_meters(redd, start_date, end_date,
                                                            building=building,
                                                            sample_period=sample_period)

        loguru.logger.debug(f"Length of data of all loaded meters {len(all_df)}")
        all_df, label2id = normalize_columns(all_df, metergroup, appliances)
        loguru.logger.debug(f"Length of data of all loaded meters {len(all_df)}")
        info('Meters that have been loaded (all_df.columns):\n' + str(all_df.columns))
        return all_df, metergroup, label2id


def get_no_samples_per_min():
    return 60/6

def get_no_samples_per_hour():
    return get_no_samples_per_min() * 60

def get_no_samples_per_day():
    return get_no_samples_per_hour() * 24

def get_window(dt: TimeSeriesLength) -> int:
    choices = {TimeSeriesLength.WINDOW_SAMPLE_PERIOD: 1,
              TimeSeriesLength.WINDOW_1_MIN: get_no_samples_per_min(),
              TimeSeriesLength.WINDOW_5_MINS: get_no_samples_per_min() * 5,
              TimeSeriesLength.WINDOW_10_MINS: get_no_samples_per_min() * 10,
              TimeSeriesLength.WINDOW_30_MINS: get_no_samples_per_min() * 30,
              TimeSeriesLength.WINDOW_1_HOUR: get_no_samples_per_hour(),
              TimeSeriesLength.WINDOW_2_HOURS: get_no_samples_per_hour() * 2,
              TimeSeriesLength.WINDOW_4_HOURS: get_no_samples_per_hour() * 4,
              TimeSeriesLength.WINDOW_8_HOURS: get_no_samples_per_hour() * 8,
              TimeSeriesLength.WINDOW_1_DAY: get_no_samples_per_day(),
              TimeSeriesLength.WINDOW_1_WEEK: get_no_samples_per_day() * 7
              }
    return int(choices.get(dt, 1))



    
# def transform(series: np.ndarray, sample_period: int = 6, dimension: int = 6, delay_in_seconds: int = 30) -> list:
#     delay_items = int(delay_in_seconds / sample_period)
#     window_size = delay_items * dimension
#     num_of_segments = int(len(series)/ window_size)
#     delay_embeddings = []
#     for i in range(num_of_segments):
#         segment = series[i * window_size:(i+1) * window_size]
#         embedding = takens_embedding(segment, delay_items, dimension)
#         delay_embeddings.append(embedding)
#     return delay_embeddings



def approximate(series_in_segments: np.ndarray, sample_period: int = 6, window: int = 1, should_fit = False, dimension: int = 6, delay_in_seconds: int = 30) -> np.ndarray:
    """
        The time series is given as segments. For each segment we extract the delay embeddings.
        """
    delay_items = int(delay_in_seconds / sample_period)
    window_size = delay_items * dimension
    array_info(series_in_segments)
    if window_size > len(series_in_segments[0]):
        raise Exception(f'Not enough data for the given delay({delay_in_seconds} seconds) and dimension ({dimension}).'
                       f'\ndelayitems * dimension > len(data): {window_size} > {len(series_in_segments[0])}')
    if window_size == len(series_in_segments[0]):
        info(f"delay embeddings equavelent to the length of each segment"
            f"{window_size} == {len(series_in_segments[0])}")
    delay_embeddings = []
    for segment in series_in_segments:
        embedding = takens_embedding(segment, delay_items, dimension)
        delay_embeddings.append(embedding)
    return np.asarray(delay_embeddings)

        
def get_multilabels(labels_df: DataFrame, appliances: List = None) -> DataFrame:
    if appliances is None:
        return labels_df
    else:
        return labels_df[appliances]

def get_site_meter_data(df: DataFrame) -> np.ndarray:
    """
        Get the data of the site meter from the given DataFrame.
        Args:
            df (DataFrame): A DataFrame containing energy data with columns corresponding to different meters.

        Returns:
            The site meter data as an array (ndarray).
        """
    debug('get_site_meter_data')
    debug(f'dataframe columns: {df.columns}')
    for col in df.columns:
        if SITE_METER in col:
            return df[col].values
    raise NoSiteMeterException("Couldn't find site meter")
        
        
def get_features(data_df: DataFrame) -> List:
    data = get_site_meter_data(data_df)
    debug(f'type of data {type(data)}')
    #data = transform(data)
    return data


"""
reduce_dimension
        It uses the method approximate of the TimeSeriesTransformer in order to achieve dimensionality reduction.
        Args:
            data_in_batches (ndarray): The data of the time series separated in batches.
            window (int): The size of the sub-segments of the given time series.
                This is not supported by all algorithms.
            target (ndarray): The labels that correspond to the given data in batches.
            should_fit (bool): True if it is supported by the algorithm of the specified time series representation.
        Returns:
            The shortened time series as an array (ndarray).

        """
def reduce_dimensions(data_in_batches: np.ndarray, window: int, target: np.ndarray,should_fit: bool = False):
    squeezed_seq = approximate(data_in_batches, window, target, should_fit)
    return squeezed_seq

def bucketize_data(data: np.ndarray, window: int) -> np.ndarray:
    """
    It segments the time series grouping it into batches. Its segment is of size equal to the window.
    Args:
        data (ndarray): The given time series.
        window (int): The size of the segments.

    Returns:

    """
    debug('bucketize_data: Initial shape {}'.format(data.shape))
    n_dims = len(data.shape)
    debug(f'n_dims = {n_dims}')
    if n_dims == 1:
        seq_in_batches = np.reshape(data, (int(len(data) / window), window))
    elif n_dims == 2:
        seq_in_batches = np.reshape(data, (int(len(data) / window), window, data.shape[1]))
    else:
        raise Exception('Invalid number of dimensions {}.'.format(n_dims))
    debug('bucketize_data: Shape in batches: {}'.format(seq_in_batches.shape))
    return seq_in_batches

def bucketize_target(target: np.ndarray, window: int) -> np.ndarray:
    """
    Creates target data according to the lenght of the window of the segmented data.
    Args:
        target (ndarray): Target data with the original size.
        window (int): The length of window that will be used to create the corresponding labels.
    Returns:
        The target data for the new bucketized time series.
    """
    target_in_batches = bucketize_data(target, window)
    any_multilabel = np.any(target_in_batches, axis=1)
    debug('bucketize_target: Shape of array in windows: {}'.format(target_in_batches.shape))
    debug('bucketize_target: Shape of array after merging windows: {}'.format(any_multilabel.shape))
    return any_multilabel


def preprocess(data_df, labels_df, appliances,should_fit: bool = True):
    start_time = time.time()
    data = get_features(data_df)
    get_features_time = time.time() - start_time
    timing(f"get features time {get_features_time}")
    debug(f"Features \n {data[:10]}")
    target = get_multilabels(labels_df,appliances)
    target = np.array(target.values)
    debug(f"Target \n {target[:10]}")
    window = get_window(cur_window)
    rem = len(data) % window
    if rem>0:
        data = data[:-rem]
        target = target[:-rem]
    target = bucketize_target(target, window)
    data = bucketize_data(data, window)
    
    start_time = time.time()
    data = reduce_dimensions(data,window,target,should_fit)
    reduce_dimensions_time = time.time() - start_time
    timing(f"reduce dimensions time {reduce_dimensions_time}")
    
    return data, target
'''

EXPERIMENT


'''



def setup_train_data(datasource, building: int, year: str, start_date: str, end_date: str, sample_period: int, appliances: List):
    train_df, train_metergroup,train_label2id = setup_one_building(appliances,datasource,building,start_date,end_date,sample_period)
    train_labels_df = create_multilabels_from_meters(train_df, train_metergroup,train_label2id)
    return train_df, train_labels_df


def setup_test_data(datasource, building: int, year: str, start_date: str, end_date: str, sample_period: int, appliances: List):
    test_df, test_metergroup,test_label2id = setup_one_building(appliances,datasource,building,start_date,end_date,sample_period)
    test_labels_df = create_multilabels_from_meters(test_df, test_metergroup,test_label2id)
    return test_df, test_labels_df



def train(appliances: list, train_df,train_labels_df, raw_data: bool = False):
    info("Preprocessing before training...")
    start_time = time.time()
    data, target = preprocess(train_df, train_labels_df, appliances)
    preprocess_time = time.time() - start_time
    timing(f"preprocess time {preprocess_time}")
    
    if len(data.shape) == 3:
        data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))
        
        info("Training...")
        start_time = time.time()
        print(data[:10])
        mlknn.fit(data,target)
        rakel.fit(data,target)
        fit_time = time.time() - start_time
        timing(f"fit time {fit_time}")
        return preprocess_time, fit_time
    
def test(appliances: list, test_df, test_labels_df, raw_data: bool = False):
    if test_df is None or test_labels_df is None:
        raise(Exception('Test data or test target is None'))
    info("Preprocessing before testing...")
    start_time = time.time()
    data, target = preprocess(test_df, test_labels_df, appliances)
    preprocess_time = time.time() - start_time
    timing(f"preprocess time {preprocess_time}")
    if len(data.shape) == 3:
        data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))
    info("Testing...")

    start_time = time.time()
    predictions = mlknn.predict(data)
    predictions_time = time.time() - start_time
    timing(f"predictions time {predictions_time}")
    info('/---------------MLkNN------------------/')
    micro = f1_score(target, predictions, average='micro')
    macro = f1_score(target, predictions, average='macro')
    info('F1 macro {}'.format(macro))
    info('F1 micro {}'.format(micro))
    
    start_time = time.time()
    predictions = rakel.predict(data)
    predictions_time = time.time() - start_time
    timing(f"predictions time {predictions_time}")
    info('/---------------Rakel------------------/')
    micro = f1_score(target, predictions, average='micro')
    macro = f1_score(target, predictions, average='macro')
    info('F1 macro {}'.format(macro))
    info('F1 micro {}'.format(micro))
    #report = classification_report(target, predictions, target_names=appliances, output_dict=True)
        # confusion_matrix = multilabel_confusion_matrix(y_true=target, y_pred=predictions.toarray())
        # confusion_matrix = None
    




'''
year = '2011'
month_end = '8'
month_start = '1'
end_date = "{}-30-{}".format(month_end, year)
start_date = "{}-1-{}".format(month_start, year)
sample_period = 6
df_mains, metergroup = read_REDD(redd, start=None, end=None, sample_period=sample_period, building=1)
# print(df_mains[(1, 1, 'REDD')].values)
print(df_mains.describe())
figure = df_mains.plot().get_figure()
'''
redd = DataSet('redd.h5')
APPLIANCES_REDD_BUILDING_1 = ['electric oven', 'fridge', 'microwave', 'washer dryer', 'unknown', 'sockets', 'light']
APPLIANCES_REDD_BUILDING_3 = ['electric furnace', 'CE appliance', 'microwave', 'washer dryer', 'unknown', 'sockets']
building = 1
sample_period = 6

redd3_train_year_start = '2011'
redd3_train_year_end = '2011'
redd3_train_month_end = '4'
redd3_train_month_start = '4'
redd3_train_end_date = "{}-30-{}".format(redd3_train_month_end, redd3_train_year_end)
redd3_train_start_date = "{}-16-{}".format(redd3_train_month_start, redd3_train_year_start)

redd3_test_year_start = '2011'
redd3_test_year_end = '2011'
redd3_test_month_end = '5'
redd3_test_month_start = '5'
redd3_test_end_date = "{}-30-{}".format(redd3_test_month_end, redd3_test_year_end)
redd3_test_start_date = "{}-17-{}".format(redd3_test_month_start, redd3_test_year_start)

redd1_train_year_start = '2011'
redd1_train_year_end = '2011'
redd1_train_month_end = '5'
redd1_train_month_start = '4'
redd1_train_end_date = "{}-17-{}".format(redd1_train_month_end, redd1_train_year_end)
redd1_train_start_date = "{}-18-{}".format(redd1_train_month_start, redd1_train_year_start)

redd1_test_year_start = '2011'
redd1_test_year_end = '2011'
redd1_test_month_end = '5'
redd1_test_month_start = '5'
redd1_test_end_date = "{}-25-{}".format(redd1_test_month_end, redd1_test_year_end)
redd1_test_start_date = "{}-18-{}".format(redd1_test_month_start, redd1_test_year_start)

cur_window = TimeSeriesLength.WINDOW_1_DAY

def main():

    train_df, train_labels_df = setup_train_data(redd, building, redd1_train_year_start, redd1_train_start_date, redd1_train_end_date, sample_period, APPLIANCES_REDD_BUILDING_1) 
    test_df, test_labels_df = setup_test_data(redd, building, redd1_test_year_end, redd1_test_start_date, redd1_test_end_date, sample_period, APPLIANCES_REDD_BUILDING_1) 
    train(APPLIANCES_REDD_BUILDING_1, train_df, train_labels_df)
    test(APPLIANCES_REDD_BUILDING_1,test_df,test_labels_df)

    
    
main()
