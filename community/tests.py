from django.test import TestCase

# Create your tests here.
from views import *
df = read_log('con050217.csv', '2018-05-02T17:00', '2018-05-02T17:01')
df = wash_log(df)
df = partition_entities(df)
df .to_csv('partition_node.csv', index=False)
df = partition_links(df)
df.to_csv('partition_link.csv', index=False)
df = exchange_fields(df)
df .to_csv('leader_at2.csv', index=False)
df = group_and_cluster(df)
df.to_csv('community_result.csv', index=True)
df = post_processing(df)
df.to_csv('post_processed_result.csv', index=False)
