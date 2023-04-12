import argparse
import pyarrow.parquet as pq
import sqlite3
import pandas as pd
import sqlalchemy
from tqdm import tqdm
import os
from typing import Any, Dict, List, Optional
import numpy as np

from graphnet.data.parquet.parallel_parquet_train_dataset import build_geometry_table
from graphnet.data.sqlite.sqlite_utilities import create_table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_data_path', type=str, required=True)
    parser.add_argument('--database_path', type=str, required=True)
    parser.add_argument('--input_data_folder', type=str, required=True)
    parser.add_argument('--geometry_table_path', type=str, required=True)
    parser.add_argument('--batch_ids', type=int, nargs='*', required=False, default=None)
    return parser.parse_args()


def load_input(meta_batch: pd.DataFrame, input_data_folder: str, geometry_table: pd.DataFrame) -> pd.DataFrame:
        """
        Will load the corresponding detector readings associated with the meta data batch.
        """
        batch_id = pd.unique(meta_batch['batch_id'])

        assert len(batch_id) == 1, "contains multiple batch_ids. Did you set the batch_size correctly?"
        
        detector_readings = pd.read_parquet(path = f'{input_data_folder}/batch_{batch_id[0]}.parquet')
        sensor_positions = geometry_table.loc[
            detector_readings['sensor_id'], 
            ['x', 'y', 'z', 'sensor_type', 'relative_qe']
        ]
        sensor_positions.index = detector_readings.index

        for column in sensor_positions.columns:
            if column not in detector_readings.columns:
                detector_readings[column] = sensor_positions[column]

        detector_readings['auxiliary'] = detector_readings['auxiliary'].replace({True: 1, False: 0})
        return detector_readings.reset_index()

def add_to_table(database_path: str,
                      df: pd.DataFrame,
                      table_name:  str,
                      is_primary_key: bool,
                      ) -> None:
    """Writes meta data to sqlite table. 

    Args:
        database_path (str): the path to the database file.
        df (pd.DataFrame): the dataframe that is being written to table.
        table_name (str, optional): The name of the meta table. Defaults to 'meta_table'.
        is_primary_key(bool): Must be True if each row of df corresponds to a unique event_id. Defaults to False.
    """
    try:
        create_table(   columns=  df.columns,
                        database_path = database_path, 
                        table_name = table_name,
                        integer_primary_key= is_primary_key,
                        index_column = 'event_id')
    except sqlite3.OperationalError as e:
        if 'already exists' in str(e):
            pass
        else:
            raise e
    engine = sqlalchemy.create_engine("sqlite:///" + database_path)
    df.to_sql(table_name, con=engine, index=False, if_exists="append", chunksize = 200000)
    engine.dispose()
    return

def convert_to_sqlite(meta_data_path: str,
                      database_path: str,
                      input_data_folder: str,
                      geometry_table: pd.DataFrame,
                      batch_size: int = 200000,
                      batch_ids: Optional[List[int]] = None,) -> None:
    """Converts a selection of the Competition's parquet files to a single sqlite database.

    Args:
        meta_data_path (str): Path to the meta data file.
        batch_size (int): the number of rows extracted from meta data file at a time. Keep low for memory efficiency.
        database_path (str): path to database. E.g. '/my_folder/data/my_new_database.db'
        input_data_folder (str): folder containing the parquet input files.
        batch_ids (List[int]): The batch_ids you want converted. Defaults to None (all batches will be converted)
    """
    if batch_ids is None:
        batch_ids = np.arange(1,661,1).tolist()
    else:
        assert isinstance(batch_ids,list), "Variable 'batch_ids' must be list."
    if not database_path.endswith('.db'):
        database_path = database_path+'.db'
    meta_data_iter = pq.ParquetFile(meta_data_path).iter_batches(batch_size = batch_size)
    batch_id = 1
    converted_batches = []
    progress_bar = tqdm(total = len(batch_ids))
    for meta_data_batch in meta_data_iter:
        if batch_id in batch_ids:
            meta_data_batch  = meta_data_batch.to_pandas()
            add_to_table(database_path = database_path,
                        df = meta_data_batch,
                        table_name='meta_table',
                        is_primary_key= True)
            pulses = load_input(meta_batch=meta_data_batch, input_data_folder= input_data_folder, geometry_table=geometry_table)
            del meta_data_batch # memory
            add_to_table(database_path = database_path,
                        df = pulses,
                        table_name='pulse_table',
                        is_primary_key= False)
            del pulses # memory
            progress_bar.update(1)
            converted_batches.append(batch_id)
        batch_id +=1
        if len(batch_ids) == len(converted_batches):
            break
    progress_bar.close()
    del meta_data_iter # memory
    print(f'Conversion Complete!. Database available at\n {database_path}')


if __name__ == '__main__':
    args = parse_args()

    geometry = build_geometry_table(args.geometry_table_path).to_pandas()

    convert_to_sqlite(
        args.meta_data_path,
        database_path=args.database_path,
        input_data_folder=args.input_data_folder,
        geometry_table=geometry,
        batch_ids=args.batch_ids,
    )

# [
#     494, 460, 14, 637, 352, 384, 52, 127, 535, 577, 
#     405, 312, 140, 170, 655, 476, 575, 76, 592, 298, 
#     401, 541, 596, 588, 206, 271, 474, 315, 48, 409, 
#     526, 103, 517, 64, 386, 200, 50, 452, 159, 142, 
#     529, 175, 419, 243, 301, 578, 75, 362, 620, 428, 
#     590, 446, 547, 627, 413, 574, 252, 625, 85, 496, 
#     210, 415, 233, 162, 626, 453
# ]  # random shuffled with seed 0