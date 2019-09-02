from django.test import TestCase

# Create your tests here.
from views import *
for date in range(19, 31):
    for hour in range(0, 24):
        print(date, hour)
        start_date = date
        end_date = date
        start_hour=hour
        end_hour = hour+1
        if hour == 23:
            end_date += 1
            end_hour = 0
        try:
            df = read_log('sselected_2018-01-{}.csv'.format(date), '2018-01-{}T{}:00'.format(start_date,start_hour),
                          '2018-01-{}T{}:00'.format(end_date,end_hour))
            df = wash_log(df)
            df = partition_entities(df)
            df = partition_links(df)
            df = exchange_fields(df)
            df = group_and_cluster(df)
            df = encode_tag(df)
            df.to_csv('./jan/{}T{}.csv'.format(start_date,start_hour), index=False,encoding='gbk')
        except Exception as e:
            print(e)
            pass
