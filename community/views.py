from __future__ import unicode_literals
import math

from django.http import HttpResponse
from django.template import loader
from pyecharts import Line3D, Graph
from django.shortcuts import render, get_object_or_404
import os

REMOTE_HOST = "https://pyecharts.github.io/assets/js"
DATA_SET_DIR = r"E:\mysite\community\static\comunity\dataset"


def index(request):
    g = get_graph()
    context = dict(
        myechart=g.render_embed(),
        host=REMOTE_HOST,
        script_list=g.get_js_dependencies(),
        files=os.listdir(DATA_SET_DIR)
    )
    return render(request, 'community/pyecharts.html', context)


def get_graph():
    data = [('1', '2', '0'),
            ('1', '3', '1'),
            ('2', '4', '1')]

    nodes = []
    links = []
    exist_nodes = set()

    for source, target, tag in data:
        if source not in exist_nodes:
            nodes.append({'name': source, 'category': int(tag), 'symbolSize': 10})
            exist_nodes.add(source)
        if target not in exist_nodes:
            nodes.append({'name': target, 'category': int(tag), 'symbolSize': 10})
            exist_nodes.add(target)

        links.append({'source': source, 'target': target})

    g = Graph(title="拓扑结构", subtitle='xx-xx', width=1200, height=500)
    g.add("", nodes, links, categories=[n['category'] for n in nodes])
    return g


def discover(request):
    g = get_graph()
    context = dict(
        myechart=g.render_embed(),
        host=REMOTE_HOST,
        script_list=g.get_js_dependencies(),
        files=os.listdir(DATA_SET_DIR)
    )
    from pprint import pprint
    pprint(request.POST)
    return render(request,'community/pyecharts.html',context)
