import pandas as pd
import sys
from argparse import ArgumentParser
import datetime
import pytz

def parse_args():
    argparser = ArgumentParser()

    argparser.add_argument('epa_file')
    argparser.add_argument('--max', action='store_true')

    return argparser.parse_args()


if __name__ == "__main__":
    args = parse_args()

  #  data = pd.read_csv(sys.stdin, index_col='unixtime')
    data = pd.read_csv('Data5Second/donovan_Round1/bd17_copy.csv', index_col='unixtime')
    data.index = pd.to_datetime(data.index, unit='s')

    groups = data.groupby(lambda x: "%u/%u/%u %u:%u:00" % (x.year, x.month, x.day, x.hour, x.minute))
    
    print('finding mean')
    if args.max:
        minute_data = groups.max()
    else:
        minute_data = groups.mean()
   

    minute_data.index = pd.to_datetime(minute_data.index).tz_localize('UTC').tz_convert('US/Pacific') 
    #minute_data.index = pd.to_datetime(minute_data.index)
    minute_data = minute_data.sort_index()

    epa_data = pd.read_csv(args.epa_file, index_col='datetime') 
    #epa_data.index = pd.to_datetime(epa_data.index)
    epa_data.index = pd.to_datetime(epa_data.index).tz_localize('US/Pacific')
  
    #print(minute_data)
    #print(epa_data)

    merged_data = minute_data.merge(epa_data, how='outer', left_index=True, right_index=True).dropna()

    print(merged_data.to_csv("hello.csv", index_label='datetime'))
