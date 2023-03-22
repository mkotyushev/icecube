import argparse
import numpy as np
import pandas as pd
from icecube_utils import (
    inference, 
    load_pretrained_model
)
from train_large import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size_multiplier', type=int, default=1)
    parser.add_argument('--test-db-path', type=str, default='/kaggle/working/test_database.db')
    parser.add_argument('--state_dict_path', type=str, default='/kaggle/input/icecube-model/state_dict.pth')
    return parser.parse_args()


def prepare_dataframe(df, angle_post_fix = '_reco', vec_post_fix = '') -> pd.DataFrame:
    r = np.sqrt(df['direction_x'+ vec_post_fix]**2 + df['direction_y'+ vec_post_fix]**2 + df['direction_z' + vec_post_fix]**2)
    df['zenith' + angle_post_fix] = np.arccos(df['direction_z'+ vec_post_fix]/r)
    df['azimuth'+ angle_post_fix] = np.arctan2(df['direction_y'+ vec_post_fix],df['direction_x' + vec_post_fix]) #np.sign(results['true_y'])*np.arccos((results['true_x'])/(np.sqrt(results['true_x']**2 + results['true_y']**2)))
    df['azimuth'+ angle_post_fix][df['azimuth'  + angle_post_fix]<0] = df['azimuth'  + angle_post_fix][df['azimuth'  +  angle_post_fix]<0] + 2*np.pi 

    drop_these_columns = []
    for column in df.columns:
        if column not in ['event_id', 'zenith', 'azimuth']:
            drop_these_columns.append(column)
    return df.drop(columns = drop_these_columns).iloc[:,[0,2,1]].set_index('event_id')


def main(args):
    config['train_transforms'] = []
    config['dynedge']['dynedge_layer_sizes'] = [
        (x * args.size_multiplier, y * args.size_multiplier) 
        for x, y in [(128, 256), (336, 256), (336, 256), (336, 256)]
    ]
    config['batch_size'] = 256
    config['truth'] = ['zenith', 'azimuth']
    config['target'] = 'direction'
    config['path'] = '/kaggle/working/test_database.db'
    config['inference_database_path'] = args.test_db_path
    config['bias'] = True
    config['fit']['distribution_strategy'] = 'ddp'

    model = load_pretrained_model(
        config=config, 
        path=args.state_dict_path,
        return_train_dataloader=False,
    )

    df = inference(
        model.cuda(), 
        config
    )

    df = prepare_dataframe(df, angle_post_fix='', vec_post_fix='')
    df = df.sort_values(by='event_id')
    df.to_csv('/kaggle/working/submission.csv')


if __name__ == '__main__':
    args = parse_args()
    main(args)
