from .load_map import CorpusLoader
from . import app
import json
import requests
from datetime import datetime
from pathlib import Path
import re
import networkx as nx



class Centrality:
    @staticmethod
    def get_nodeset_path(nodeset_id):
        corpus_name = 'US2016tv'
        directory_path = 'examples/'
        node_path = directory_path + 'nodeset' + nodeset_id + '.json'
        return node_path

    @staticmethod
    def get_svg_path(nodeset_id):
        #corpus_name = 'US2016tv'
        directory_path = 'examples/'
        node_path = directory_path + nodeset_id + '.svg'
        return node_path

    @staticmethod
    def create_svg_url(nodeset_id, isMap):

        if isMap:
            return 'http://www.aifdb.org/diagram/svg/' + nodeset_id
        else:
            return 'http://corpora.aifdb.org/' + nodeset_id + '/svg/'
        return node_path

    @staticmethod
    def create_json_url(nodeset_id, isMap):

        if isMap:
            return 'http://www.aifdb.org/json/' + nodeset_id
        else:
            return 'http://corpora.aifdb.org/' + nodeset_id + '/json/'
        return node_path

    @staticmethod
    def get_graph(node_path):
        corpus_loader = CorpusLoader()
        try:
            with app.open_resource(node_path) as json_data:
                graph = corpus_loader.parse_json(json.load(json_data))
        except(IOError):
            print('File was not found:')
            print(node_path)

        return graph

    @staticmethod
    def get_graph_url(node_path):
        corpus_loader = CorpusLoader()
        print(node_path)
        try:
            jsn_string = requests.get(node_path).text
            strng_ind = jsn_string.index('{')
            n_string = jsn_string[strng_ind:]
            dta = json.loads(n_string)
            graph = corpus_loader.parse_json(dta)
        except(IOError):
            print('File was not found:')
            print(node_path)

        return graph

    @staticmethod
    def remove_redundant_nodes(graph):

        node_types=nx.get_node_attributes(graph,'type')

        nodes_to_remove =  [x for x,y in graph.nodes(data=True) if y['type']=='TA' or y['type']=='L' or y['type']=='YA']

        graph.remove_nodes_from(nodes_to_remove)

        return graph
    @staticmethod
    def remove_iso_nodes(graph):
        graph.remove_nodes_from(list(nx.isolates(graph)))
        return graph

    @staticmethod
    def get_eigen_centrality(graph):
        try:
            cent = nx.eigenvector_centrality_numpy(graph)
        except:
            cent = nx.degree_centrality(graph)

        nx.set_node_attributes(graph, cent, 'central')
        i_nodes =  [(x,y['central'],y['text']) for x,y in graph.nodes(data=True) if y['type']=='I']
        return i_nodes

    @staticmethod
    def sort_by_centrality(i_nodes):
        sorted_by_second = sorted(i_nodes, key=lambda tup: tup[1])
        ordered_ids = [(i[0],i[2]) for i in sorted_by_second]

        return ordered_ids

    @staticmethod
    def list_nodes(graph):
        return list(graph)

    @staticmethod
    def get_s_node_list(graph):
        s_nodes =  [x for x,y in graph.nodes(data=True) if y['type']=='MA' or y['type']=='RA' or y['type']=='CA' or y['type']=='PA']
        return s_nodes


    @staticmethod
    def get_l_node_list(graph):
        l_nodes =  [(x,y['text']) for x,y in graph.nodes(data=True) if y['type']=='L']
        return l_nodes

    @staticmethod
    def get_divergent_nodes(graph):
        list_of_nodes = []

        for v in list(graph.nodes):
            node_pres = []
            node_pres = list(graph.successors(v))
            if len(node_pres) > 1:
                list_of_nodes.append(v)
        return list_of_nodes

    @staticmethod
    def get_loc_prop_pair(graph):
        i_node_ids =  [x for x,y in graph.nodes(data=True) if y['type']=='I']
        locution_prop_pair = []
        for node_id in i_node_ids:
            preds = list(graph.predecessors(node_id))
            for pred in preds:
                node_type=graph.nodes[pred]['type']
                node_text = graph.nodes[pred]['text']

                if node_type == 'YA' and node_text != 'Agreeing':
                    ya_preds = list(graph.predecessors(pred))
                    for ya_pred in ya_preds:
                        pred_node_type=graph.nodes[ya_pred]['type']
                        pred_node_text=graph.nodes[ya_pred]['text']

                        if pred_node_type == 'L':
                            locution_prop_pair.append((ya_pred, node_id))
        return locution_prop_pair

    @staticmethod
    def get_child_edges(graph):
        list_of_nodes = []
        list_of_edges = []

        for v in list(graph.nodes):
            node_pres = []
            node_pres = list(nx.ancestors(graph, v))
            list_of_nodes.append((v, node_pres))
            edges = []
            edges = list(nx.edge_dfs(graph,v, orientation='reverse'))
            res_list = []
            res_list = [(x[0], x[1]) for x in edges]
            list_of_edges.append((v, res_list))

        return list_of_nodes, list_of_edges
    @staticmethod
    def get_top_nodes_combined(node_list):
        all_nodes = []
        G = nx.DiGraph()
        for node in node_list:
            dir_path = 'http://www.aifdb.org/json/' + str(node)
            g1 = get_graph_url(dir_path)

            G = nx.compose(G,g1)

        g = remove_redundant_nodes(G)
        i_nodes = get_eigen_centrality(g)
        sorted_nodes = sort_by_centrality(i_nodes)
        #print(len(i_nodes))
        #ten_percent = 0.1 * len(i_nodes)
        #top_n = sorted_nodes[:int(ten_percent)]
        all_nodes.extend(sorted_nodes)

        ten_percent = 0.05 * len(all_nodes)

        return all_nodes[:int(ten_percent)]
