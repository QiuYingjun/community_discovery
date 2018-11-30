# coding:utf8
from __future__ import unicode_literals
from pyecharts import Graph, Bar
from django.shortcuts import render
import time
import os
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from .models import Result, Community
import numpy as np
import django.utils.timezone as timezone
from collections import Counter

PROJECT_DIR = os.path.dirname(__file__)
DATA_SET_DIR = os.path.join(PROJECT_DIR, "static/community/dataset")
RESULT_DIR = os.path.join(PROJECT_DIR, "static/community/result")
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# 至少有这么IP对才会做聚类操作
LEAST_IP_PAIR_FOR_CLUSTERING = 3


def index(request):
    """
    使用最近一次发现结果渲染页面

    :param request:
    :return:
    """
    # if Result.objects.all():
    #     last_result = Result.objects.latest('result_time')
    #     df, last_result, communities = get_result_df(log_filename=last_result.log_filename,
    #                                                  start_time=last_result.start_time,
    #                                                  end_time=last_result.end_time,
    #                                                  smallest_size=last_result.smallest_size)
    #     g = get_graph(df)
    # else:
    print('index')
    last_result = Result()
    g = get_graph(pd.DataFrame())
    communities = {}

    context = dict(
        graph=g.render_embed(),
        files=os.listdir(DATA_SET_DIR),
        algorithms=['lpa','sdsf', 'abc'],
        last_result=last_result,
        communities=communities
    )

    return render(request, 'community/discover2.html', context)


def get_graph(df):
    """
    根据发现结果df生成关系图

    :param df: df
    :return:
    """
    print('get_graph')
    links = []
    exist_nodes = {}

    for i in df.index:
        source = df.loc[i, 'ip1']
        target = df.loc[i, 'ip2']
        if source not in exist_nodes:
            exist_nodes[source] = []
        if target not in exist_nodes:
            exist_nodes[target] = []
        exist_nodes[source].append(int(df.loc[i, 'community_tag']))
        exist_nodes[target].append(int(df.loc[i, 'community_tag']))
        links.append({'source': source, 'target': target})
    nodes = []
    categories = []
    for node, cato_list in exist_nodes.items():
        # 取众数
        cat = Counter(cato_list).most_common(1)[0][0]
        nodes.append({'name': node, 'symbolSize': 10, 'value': cat})
        categories.append(cat)
    categories = LabelEncoder().fit_transform(categories)
    for node, cat in zip(nodes, categories):
        node['category'] = cat
    g = Graph(title="拓扑结构", subtitle='IP:{} Links:{}'.format(len(nodes), len(links)), width=1200, height=500)
    g.add("", nodes, links, layout='circular', categories=list(categories))
    return g


def get_result_df(log_filename, start_time, end_time, smallest_size):
    """
    根据指定参数得到发现结果表，如果已有则读取，没有则计算得到

    :param log_filename:
    :param start_time:
    :param end_time:
    :param smallest_size:
    :return:
    """
    print('get_result_df')
    result_filename = '{}_{}_{}.csv'.format(log_filename, start_time, end_time).replace(':', '')
    if os.path.exists(os.path.join(RESULT_DIR, result_filename)):
        df = pd.read_csv(os.path.join(RESULT_DIR, result_filename))
        # 移除小社团
        df = remove_small_community(df, smallest_size=smallest_size)

        result = Result.objects.get(result_filename__exact=result_filename)
        result.result_time = timezone.now()
        result.smallest_size = smallest_size
        result.save()
    else:
        # 读入
        df = read_log(log_filename, start_time, end_time)
        df.to_csv(os.path.join(RESULT_DIR, '_df1.csv'), index=False)
        # 清洗
        df = wash_log(df)
        df.to_csv(os.path.join(RESULT_DIR, '_df2.csv'), index=False)
        # 节点区分
        df = partition_entities(df)
        df.to_csv(os.path.join(RESULT_DIR, '_df3.csv'), index=False)
        # 边分区
        df = partition_links(df)
        df.to_csv(os.path.join(RESULT_DIR, '_df4.csv'), index=False)
        # 交换IP使领袖节点在同一列
        df = exchange_fields(df)
        df.to_csv(os.path.join(RESULT_DIR, '_df5.csv'), index=False)
        # 对拓扑相似的边聚类
        df = group_and_cluster(df)
        df.to_csv(os.path.join(RESULT_DIR, '_df6.csv'), index=False)
        # 标签重编码
        df = encode_tag(df)
        df.to_csv(os.path.join(RESULT_DIR, result_filename), index=False)
        # 移除小社团
        df = remove_small_community(df, smallest_size=smallest_size)

        result = Result()
        result.result_time = timezone.now()
        result.smallest_size = smallest_size
        result.end_time = end_time
        result.start_time = start_time
        result.log_filename = log_filename
        result.community_counts = len(set(df['community_tag']))
        result.ip_counts = len(set(df['ip1']) | set(df['ip2']))
        result.result_filename = result_filename
        result.save()

    communities = set()
    for community_tag, community_table in df.groupby('community_tag'):
        community = Community(community_tag=community_tag)
        community.result = result
        community.ip_counts = len(set(community_table['ip1']) | set(community_table['ip2']))
        community.link_counts = community_table.shape[0]
        community.leader_ip = community_table['ip2'].mode()[0]
        # community.save()
        communities.add(community)
    communities = sorted(communities, key=lambda c: c.ip_counts, reverse=True)

    return df, result, communities


def discover(request):
    """
    展示发现结果

    :param request:
    :return: 渲染后的页面
    """
    print('discover')
    log_filename = request.POST['filename']
    start_time = request.POST['start_time']
    end_time = request.POST['end_time']
    smallest_size = request.POST['smallest_size']
    df, last_result, communities = get_result_df(log_filename=log_filename, start_time=start_time, end_time=end_time,
                                                 smallest_size=smallest_size)
    g = get_graph(df)
    context = dict(
        graph=g.render_embed(),
        files=os.listdir(DATA_SET_DIR),
        communities=communities,
        last_result=last_result,

    )
    return render(request, 'community/discover.html', context)


def get_hist(df, community_tag, feature_cols):
    """
    生成柱形图

    :param df:
    :param community_tag:
    :param feature_cols:
    :return:
    """
    # feature_cols = ['port1','port2','proto',
    #                 'pkts12_min','pkts12_mid','pkts12_max',
    #                 'pkts21_min','pkts21_mid','pkts21_max',
    #                 'pkl12_min','pkl12_mid','pkl12_max',
    #                 'pkl21_min','pkl21_mid','pkl21_max']
    print('get_hist')

    community_table = df[df['community_tag'] == community_tag].copy()[feature_cols]
    axis_min = int(community_table.values.min())
    axis_max = int(community_table.values.max())
    bar = Bar("")
    for col in feature_cols:
        values_counts = community_table[col].value_counts()
        for i in range(axis_min, axis_max + 1):
            if i not in values_counts.index:
                values_counts.loc[i] = 0
        values_counts = values_counts.sort_index()
        bar.add(col, values_counts.index, values_counts.values, legend_pos='center', legend_top='bottom')
    return bar


def detail(request, community_tag):
    """
    某个社团的详细信息

    :param request:
    :param community_tag:
    :return:
    """
    print('detail')
    last_result = Result.objects.latest('result_time')
    df, last_result, communities = get_result_df(log_filename=last_result.log_filename,
                                                 start_time=last_result.start_time,
                                                 end_time=last_result.end_time,
                                                 smallest_size=last_result.smallest_size)
    feature_groups = [
        ['port1', 'port2', 'proto'],
        ['pkts12_min', 'pkts12_mid', 'pkts12_max'],
        ['pkts21_min', 'pkts21_mid', 'pkts21_max'],
        ['pkl12_min', 'pkl12_mid', 'pkl12_max'],
        ['pkl21_min', 'pkl21_mid', 'pkl21_max']
    ]
    bars = []
    for feature_cols in feature_groups:
        bars.append(get_hist(df, community_tag, feature_cols).render_embed())
    graph = get_graph(df[df['community_tag'] == community_tag]).render_embed()
    context = dict(
        graph=graph,
        bars=bars,
        community_tag=community_tag,
        communities=communities
    )

    return render(request, 'community/detail.html', context)


def read_log(filename, start_time, end_time):
    """
    读取指定时间区间的日志

    :param filename: 文件名
    :param start_time: "2018-05-02T17:00"
    :param end_time:
    :return: DataFrame log_df
    """
    print('read_log')
    useful_columns = ['ip1', 'ip2', 'port1', 'port2', 'proto', 'pkts12', 'pkts21', 'bytes12', 'bytes21', 'etime',
                      'class', 'app']
    log_df = pd.read_csv(os.path.join(DATA_SET_DIR, filename), usecols=useful_columns)
    start_time = time.mktime(time.strptime(start_time, "%Y-%m-%dT%H:%M"))
    end_time = time.mktime(time.strptime(end_time, "%Y-%m-%dT%H:%M"))

    return log_df[log_df['etime'] >= start_time][log_df['etime'] < end_time]


def wash_log(log_df):
    """
    去重复行

    :param log_df: 原始log_df
    :return: clean_df
    """
    print('wash_log')

    clean_df = log_df.drop_duplicates(['ip1', 'ip2', 'port1', 'port2', 'proto'], keep='last')

    return clean_df


def construct_topology(clean_df):
    """
    拓扑构建
    {entity:{neighbour1,neighbour2...}}

    :param clean_df:
    :return:
    """
    print('construct_topology')
    entity_neighbours = {}
    for i in clean_df.index:
        entity1 = clean_df.at[i, 'entity1']
        entity2 = clean_df.at[i, 'entity2']
        if entity1 not in entity_neighbours:
            entity_neighbours[entity1] = set()
        entity_neighbours[entity1].add(entity2)
        if entity2 not in entity_neighbours:
            entity_neighbours[entity2] = set()
        entity_neighbours[entity2].add(entity1)
    return entity_neighbours


def distribute_num(entity_neighbours):
    """
    给节点分配编号

    :param entity_neighbours:
    :return:
    """
    print('distribute_num')
    num = 0
    entity_num = {}
    for entity1 in entity_neighbours:
        if entity1 in entity_num:
            pass
        else:
            P = entity_neighbours[entity1]
            Q = set()
            for entity2 in entity_neighbours[entity1]:
                for entity3 in entity_neighbours[entity2]:
                    Q.add(entity3)
            Q = Q - P - {entity1}

            if len(P) > len(Q):
                entity_num[entity1] = num
                for entity2 in P:
                    entity_num[entity2] = num
            num += 1
    return entity_num


def partition_entities(clean_df):
    """
    追加两列，内容为<src_ip，port>所属区号及<dst_ip, port>所属区号

    :param clean_df:
    :return:
    """
    print('partition_entities')
    clean_df['entity1'] = clean_df['ip1'] + '_' + clean_df['port1'].astype(str)
    clean_df['entity2'] = clean_df['ip2'] + '_' + clean_df['port2'].astype(str)

    entity_neighbours = construct_topology(clean_df)
    entity_num = distribute_num(entity_neighbours)
    for i in clean_df.index:
        if clean_df.at[i, 'entity1'] in entity_num:
            clean_df.at[i, 'part1'] = entity_num[clean_df.at[i, 'entity1']]
        else:
            clean_df.at[i, 'part1'] = -1

        if clean_df.at[i, 'entity2'] in entity_num:
            clean_df.at[i, 'part2'] = entity_num[clean_df.at[i, 'entity2']]
        else:
            clean_df.at[i, 'part2'] = -1
    clean_df['part1'] = clean_df['part1'].astype(int)
    clean_df['part2'] = clean_df['part2'].astype(int)
    return clean_df


def partition_links(label_entities_df):
    """
    追加一列，值为边的编号

    :param label_entities_df:
    :return:
    """
    print('partition_links')

    for i in label_entities_df.index:
        if label_entities_df.at[i, 'part1'] < label_entities_df.at[i, 'part2']:
            link_label = str(label_entities_df.at[i, 'part1']) + '_' + str(label_entities_df.at[i, 'part2'])
        else:
            link_label = str(label_entities_df.at[i, 'part2']) + '_' + str(label_entities_df.at[i, 'part1'])
        label_entities_df.at[i, 'link_label'] = link_label
    return label_entities_df


def determine_leader(label_links_df):
    """
    选领袖节点

    :param label_links_df:
    :return: {link_label:leader_ip}
    """
    print('determine_leader')
    label_leader = {}
    link_groups = label_links_df.groupby(by='link_label')
    for link_label, same_label_df in link_groups:
        ip_counts = same_label_df['entity1'].value_counts().add(same_label_df['entity2'].value_counts(), fill_value=0)
        label_leader[link_label] = ip_counts.idxmax().split('_')[0]
    return label_leader


def exchange_fields(label_links_df):
    """
    保证领袖节点处于ip2位置

    :param label_links_df:
    :return:
    """
    print('exchange_fields')
    label_links_df = label_links_df[label_links_df['link_label'] != '-1_-1']
    label_leader = determine_leader(label_links_df)

    need_exchange = label_links_df['ip1'] == label_links_df['link_label'].apply(lambda x: label_leader[x])

    label_links_df.loc[need_exchange, 'ip1'], label_links_df.loc[need_exchange, 'ip2'] = \
        label_links_df.loc[need_exchange, 'ip2'], label_links_df.loc[need_exchange, 'ip1']

    label_links_df.loc[need_exchange, 'port1'], label_links_df.loc[need_exchange, 'port2'] = \
        label_links_df.loc[need_exchange, 'port2'], label_links_df.loc[need_exchange, 'port1']

    label_links_df.loc[need_exchange, 'part1'], label_links_df.loc[need_exchange, 'part2'] = \
        label_links_df.loc[need_exchange, 'part2'], label_links_df.loc[need_exchange, 'part1']

    label_links_df.loc[need_exchange, 'entity1'], label_links_df.loc[need_exchange, 'entity2'] = \
        label_links_df.loc[need_exchange, 'entity2'], label_links_df.loc[need_exchange, 'entity1']

    label_links_df.loc[need_exchange, 'pkts12'], label_links_df.loc[need_exchange, 'pkts21'] = \
        label_links_df.loc[need_exchange, 'pkts21'], label_links_df.loc[need_exchange, 'pkts12']

    label_links_df.loc[need_exchange, 'bytes12'], label_links_df.loc[need_exchange, 'bytes21'] = \
        label_links_df.loc[need_exchange, 'bytes21'], label_links_df.loc[need_exchange, 'bytes12']

    return label_links_df


def group_and_cluster(leader2_df):
    """
    对link_label相同的流按IP对聚合

    :param leader2_df:
    :return: {link_label:feature_df}
    """
    print('group_and_cluster')

    label_group = leader2_df.groupby(by=['link_label'])
    community_result = pd.DataFrame()
    for link_label, same_label_df in label_group:
        feature_df, verification_df = extract_feature_df(same_label_df)
        if len(feature_df) < LEAST_IP_PAIR_FOR_CLUSTERING:
            feature_df['behavior_tag'] = -2
        else:
            feature_df['behavior_tag'] = cluster(feature_df)

        feature_df['topology_tag'] = link_label
        temp_df = pd.concat([feature_df, verification_df], axis=1)
        community_result = pd.concat([community_result, temp_df])
    return community_result


def extract_feature_df(same_label_df):
    print('extract_feature_df')
    feature_df = pd.DataFrame(columns=['port1', 'port2', 'proto',
                                       'pkts12_min', 'pkts12_mid', 'pkts12_max',
                                       'pkts21_min', 'pkts21_mid', 'pkts21_max',
                                       'pkl12_min', 'pkl12_mid', 'pkl12_max',
                                       'pkl21_min', 'pkl21_mid', 'pkl21_max'
                                       ])
    verification_df = pd.DataFrame(columns=['class', 'app'])
    ippair_grouped = same_label_df.groupby(by=['ip1', 'ip2'])
    for ippair, netflow_df in ippair_grouped:
        pkl12 = netflow_df['bytes12'] / netflow_df['pkts12'].replace(0, 1)
        pkl21 = netflow_df['bytes21'] / netflow_df['pkts21'].replace(0, 1)

        new_pair = pd.Series({'port1': len(netflow_df['port1'].value_counts()),
                              'port2': len(netflow_df['port2'].value_counts()),
                              'proto': netflow_df['proto'].mean(),

                              'pkts12_min': netflow_df['pkts12'].min(),
                              'pkts12_mid': netflow_df['pkts12'].median(),
                              'pkts12_max': netflow_df['pkts12'].max(),
                              'pkts21_min': netflow_df['pkts21'].min(),
                              'pkts21_mid': netflow_df['pkts21'].median(),
                              'pkts21_max': netflow_df['pkts21'].max(),

                              'pkl12_min': pkl12.min(),
                              'pkl12_mid': pkl12.median(),
                              'pkl12_max': pkl12.max(),
                              'pkl21_min': pkl21.min(),
                              'pkl21_mid': pkl21.median(),
                              'pkl21_max': pkl21.max()
                              }, name=ippair)

        feature_df = feature_df.append(new_pair)
        verification_df = verification_df.append(pd.Series({'class': netflow_df['class'].mode()[0],
                                                            'app': netflow_df['app'].mode()[0]}, name=ippair))
    return feature_df, verification_df


def normalize(X, func=None):
    print('normalize')

    if func == 'atan':
        X = np.arctan(X) * 2 / np.pi
    elif func == 'z-score':
        X = StandardScaler().fit_transform(X)
    else:
        X = MinMaxScaler().fit_transform(X)
    return X


def cluster(feature_df):
    """

    :param feature_df:
    :return:
    """
    print('cluster')
    X = feature_df.values
    X = normalize(X)
    dbscan = DBSCAN(eps=1, min_samples=2, metric='manhattan')
    return dbscan.fit_predict(X)


def encode_tag(community_result_df):
    """

    :param community_result_df:
    :param smallest_size:
    :return:
    """
    # ip列
    print('post_processing')
    ip1s = []
    ip2s = []
    for ip1, ip2 in community_result_df.index:
        ip1s.append(ip1)
        ip2s.append(ip2)
    ip1s = pd.Series(ip1s, index=community_result_df.index, name='ip1')
    ip2s = pd.Series(ip2s, index=community_result_df.index, name='ip2')
    community_result_df = pd.concat([ip1s, ip2s, community_result_df], axis=1)

    # 重分配索引
    community_result_df.index = range(len(community_result_df))

    # topology_tag和behavior_tag结合成community_tag
    tags = community_result_df['topology_tag'].astype(str) + '_' + community_result_df['behavior_tag'].astype(str)
    community_result_df['community_tag'] = pd.Series(LabelEncoder().fit_transform(tags),
                                                     index=community_result_df.index)

    return community_result_df


def give_ip_tag(community_result_df):
    print('give_ip_tag')
    # 为节点分配所属社团，用于显示颜色
    ip_belong = {}
    for i in community_result_df.index:
        ip1 = community_result_df.loc[i, 'ip1']
        ip2 = community_result_df.loc[i, 'ip2']
        community_tag = community_result_df.loc[i, 'community_tag']
        if ip1 not in ip_belong:
            ip_belong[ip1] = set()
        if ip2 not in ip_belong:
            ip_belong[ip2] = set()
        ip_belong[ip1].add(community_tag)
        ip_belong[ip2].add(community_tag)
    for ip in ip_belong:
        ip_belong[ip] = '_'.join(map(str, sorted(list(ip_belong[ip]))))
    le = LabelEncoder()
    le.fit(list(ip_belong.values()))

    tag1 = community_result_df['ip1'].apply(lambda ip: ip_belong[ip])
    community_result_df['tag1'] = le.transform(tag1)
    tag2 = community_result_df['ip2'].apply(lambda ip: ip_belong[ip])
    community_result_df['tag2'] = le.transform(tag2)
    return community_result_df


def remove_small_community(community_result_df, smallest_size=5):
    """

    :param community_result_df:
    :param smallest_size: 最小IP数
    :return:
    """
    print('remove_small_community')
    tag_ips = {}
    for i in community_result_df.index:
        tag = community_result_df.loc[i, 'community_tag']
        ip1 = community_result_df.loc[i, 'ip1']
        ip2 = community_result_df.loc[i, 'ip2']

        if tag not in tag_ips:
            tag_ips[tag] = set()
        tag_ips[tag].add(ip1)
        tag_ips[tag].add(ip2)
    tag_ips = {tag: ips for tag, ips in tag_ips.items() if len(ips) >= int(smallest_size)}
    community_result_df = community_result_df[community_result_df['community_tag'].isin(tag_ips)]
    return community_result_df
