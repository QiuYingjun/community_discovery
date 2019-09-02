import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from itertools import product, combinations
from collections import Counter


def modularity(sources, destinations, community):
    links = {}
    m = 0
    for s, d in zip(sources, destinations):
        if s not in links:
            links[s] = set()
        if d not in links:
            links[d] = set()
        links[s].add(d)
        links[d].add(s)
        m += 1
    Q = 0
    for s, d in zip(sources, destinations):
        if s in community and d in community and community[s] == community[d]:
            Q += 1 - 0.5 * len(links[s]) * len(links[d]) / m
    Q /= 2 * m
    return Q


def jakalan_discover(part1, part2):
    links = {}
    for s, d in zip(part1, part2):
        if s not in links:
            links[s] = set()
        if d not in links:
            links[d] = set()
        links[s].add(d)
        links[d].add(s)

    S = {}
    neighbours = {}
    for d in set(part2):
        for si, sj in combinations(links[d], 2):
            if si != sj:
                S[(si, sj)] = S.get((si, sj), 0) + 1
                S[(sj, si)] = S.get((sj, si), 0) + 1
                if si not in neighbours:
                    neighbours[si] = set()
                if sj not in neighbours:
                    neighbours[sj] = set()
                neighbours[si].add(sj)
                neighbours[sj].add(si)
    for i in neighbours:
        AF = {}
        for k in neighbours[i]:
            for j in neighbours[k]:
                if j != i:
                    AF[k] = AF.get(k, 0) + S.get((i, j), 0)
        if AF:
            AF_max = max(AF.values())

            for k in neighbours[i]:
                if k in AF and AF[k] != AF_max:
                    if (i, k) in S:
                        del (S[(i, k)])
                    if (k, i) in S:
                        del (S[(k, i)])
    community = {}
    ip_encoder = {}
    n = 0
    for i in neighbours:
        for k in neighbours[i]:
            if (i, k) in S:
                if k not in ip_encoder:
                    ip_encoder[k] = n
                    n += 1
                community[i] = ip_encoder[k]

    for d in part2:
        id_list = [community[s] for s in links[d] if s in community]
        if id_list:
            community[d] = Counter(id_list).most_common(1)[0][0]

    return community


def main():
    x = []
    y = []
    for i in range(12):
        x.append(i + 1)
        df = pd.read_csv('result/test_con050217.csv_2018-05-02T17{:0>2d}_2018-05-02T17{:0>2d}.csv_df4.csv'.format(i * 5,i*5+5)).head(1000)
        part1 = df['ip1'] + '_' + df['port1'].astype(str)
        part2 = df['ip2'] + '_' + df['port2'].astype(str)
        # community = jakalan_discover(part1, part2)
        community = {}
        for i in df.index:
            community['{}_{}'.format(df.loc[i, 'ip2'], df.loc[i, 'port2'])] = df.loc[i, 'part2']
            community['{}_{}'.format(df.loc[i, 'ip1'], df.loc[i, 'port1'])] = df.loc[i, 'part1']

        mod = modularity(part1, part2, community)
        print(mod)
        y.append(mod)


if __name__ == '__main__':
    # print(jakalan_discover(list('abcd1234'), list('eeeeffff')))
    main()
