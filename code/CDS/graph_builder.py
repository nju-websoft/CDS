import pandas
import sys
import os
import shutil
import json
import fcntl
import traceback
import xmltodict
import time
from rdflib.plugins.shared.jsonld import util
from rdflib import Graph
from pebble import ProcessPool

start_time, end_time = time.time(), time.time()
triple_list = []    #edge_list
term_list = []  #label_list
cnt = 0     #term_count

edge_dict, entity_dict, literal_dict = {}, {}, {}

class Builder:
    data_tuple = []
    
    def __init__(self, readpath, writepath):
      self.readpath = readpath      
      self.writepath = writepath
    
    def is_json(self, my_str):
        try:
            json.loads(my_str)
            return True
        except json.JSONDecodeError:
            return False

    def initialize(self):
        global start_time
        global triple_list, term_list, cnt
        global edge_dict, entity_dict, literal_dict
        start_time = time.time()
        triple_list = []  
        term_list = []
        cnt = 0
        edge_dict.clear()
        entity_dict.clear()
        literal_dict.clear()
    
    def get_query_content(self, file_path, query_id):
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')  # 按制表符分割每行数据
                if len(parts) == 2:
                    current_id, content = parts
                    if current_id == query_id:
                        return content
        return 'NULL'

    def read(self):
        with open(os.path.join(self.readpath, 'del_datainfo.txt'), "r") as tuplefiles:
            self.data_tuple = [line.strip().split() for line in tuplefiles]

    def create_folder(self, data_name):
        folderpath = os.path.join(self.writepath, data_name)
        flag = False
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
            flag = True
            #shutil.rmtree(folderpath) # caution this!
        return folderpath, flag

    def add_term(self, label, kind):
        global term_list, cnt
        global edge_dict, entity_dict, literal_dict

        #对label的长度做截取 要先判空
        if label is None or label == "":
            label = " "
        if len(label) > 200:
            label = label[:200]
        label = str(label).replace("\n", " ").replace("\t", " ").replace("\r", " ")

        if kind == 0: #entity
            if label == " ":
                cnt = cnt + 1
                node_id = cnt
                entity_dict[(" ", node_id)] = cnt
                term_list.append([cnt, label, kind])
            else:
                if label not in entity_dict:
                    cnt = cnt + 1
                    entity_dict[label] = cnt
                    term_list.append([cnt, label, kind])
                node_id = entity_dict[label]
        elif kind == 1: #literal   whether use dict is not determined
            cnt = cnt + 1
            node_id = cnt
            term_list.append([node_id, label, kind])
        elif kind == 2: #edge
            if label not in edge_dict:
                cnt = cnt + 1
                edge_dict[label] = cnt
                term_list.append([cnt, label, kind])
            node_id = edge_dict[label]
        return node_id

    def add_triple(self, sub, pre, obj):
          global triple_list
          triple_list.append([sub, pre, obj])

    def store_file(self, filepath, data_list:list):
        with open(filepath, 'a+', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            for data in data_list:
                f.write('\t'.join([str(item) for item in data]) + '\n')
    def save(self, FileName):
        try:
            global start_time, end_time
            end_time = time.time()
            assert(len(term_list) > 0)
            assert(len(triple_list) > 0)
            self.store_file(os.path.join(os.path.join(self.writepath, FileName), 'term.tsv'), term_list)
            self.store_file(os.path.join(os.path.join(self.writepath, FileName), 'triple.tsv'), triple_list)
            with open(os.path.join(os.path.join(self.writepath, FileName), 'time.txt'), 'w') as time_file:
                time_file.write(f"Time taken to generate term and triple files: {end_time - start_time} seconds")
        except Exception as e:
            traceback.print_exc()
    #1、对于长度过长的字符串截取前200， 2、对于行&单元格的空值未做处理

    def task_done(self, future):
        try:
            result = future.result()  # blocks until results are ready
        except TimeoutError as error:
            print("Function took longer than %d seconds" % error.args[1])
        except Exception as error:
            print("Function raised %s" % error)
            print(error.traceback)  # traceback of the function

    def handle_csv(self, FileName):
        root_path, flag = self.create_folder(FileName) # can be substituded with other name  
        if flag: # build graph
            fp = os.path.join(os.path.join(os.path.join(self.readpath, 'NTCIR'), 'data_search_e_data'), FileName)
            try:
                df = pandas.read_csv(fp, header=0, on_bad_lines='skip', low_memory=False) # on_bad_lines 可能需要被替换为更精确的检查
            except UnicodeDecodeError:
                # 如果捕获到编码异常，什么都不做
                return

            self.initialize()

            Col, Row = [], []
            headers = df.columns
            for h in headers:
                if not pandas.isna(h):
                    Col.append([self.add_term(str(h), 2), h])
            assert(len(Col) > 0)
            for index, row in df.iterrows():
                # add row term
                row_id = self.add_term(" ", 0)
                Row.append(row_id)
                # add unit term and edge
                for tp in Col:
                    col_id, col_data = tp
                    if pandas.isna(row[col_data]):
                        unit_id = self.add_term(" ", 1)
                    else:
                        unit_id = self.add_term(str(row[col_data]), 1)
                    self.add_triple(row_id, col_id, unit_id)
            assert(len(Row) > 0)
            for i in range(0, len(Row)):
                if i + 1 != len(Row):
                    edge_id = self.add_term(" ", 2)
                    self.add_triple(Row[i], edge_id, Row[i + 1])       
            self.save(FileName)
    # 需要注意空值的处理 和字符串的截断
    def dict_generator(self, data, pre_node_id = None, pre_edge_id = None):
        node_id = None
        if data is None:
            return   
        elif isinstance(data, dict):
            if len(data) != 0:
                node_id = self.add_term(" ", 0)
                for key, value in data.items():
                    edge_id = self.add_term(key, 2)
                    self.dict_generator(value, node_id, edge_id)
                        # add edge
            if pre_node_id is not None and node_id is not None:
                self.add_triple(pre_node_id, pre_edge_id, node_id)
        elif isinstance(data, list) or isinstance(data, tuple):
            if len(data) != 0:
                for value in data:
                    self.dict_generator(value, pre_node_id, pre_edge_id)
            else:
                return
            '''
            if len(data) != 0:
                node_id = self.add_term(" ", 0)
                for value in data:
                    edge_id = self.add_term(" ", 2)
                    if isinstance(value, list) or isinstance(value, tuple) or isinstance(value, dict):
                        #str_node = json.dumps(value) 
                        self.dict_generator(value, node_id, edge_id)
                    elif isinstance(value, str):
                        value = value.strip()
                        if value == "": # NULL value
                            value = " "
                        self.dict_generator(value, node_id, edge_id)
            else:
                return
            '''
        elif isinstance(data, str):
            data = data.strip()
            #bug?
            #if len(data) > 200:
                #data = data[:200]
            if data == "":
                data = " "
            node_id = self.add_term(data, 1)
            # add edge
            if pre_node_id is not None and node_id is not None:
                self.add_triple(pre_node_id, pre_edge_id, node_id)
        else: # for safety
            node_id = self.add_term(str(data), 1)
            # add edge
            if pre_node_id is not None and node_id is not None:
                self.add_triple(pre_node_id, pre_edge_id, node_id)

    def handle_json(self, FileName):
        root_path, flag = self.create_folder(FileName) # can be substituded with other name
        if flag:
            # build graph
            fp = os.path.join(os.path.join(os.path.join(self.readpath, 'NTCIR'), 'data_search_e_data'), FileName)

            self.initialize()
            with open(fp, encoding = "utf-8") as jsonfile:
                row_data = json.load(jsonfile)
                self.dict_generator(row_data)
            # need to sava graph here
            self.save(FileName)

    def handle_xls(self): # Todo
        return

    def handle_xml(self, FileName):
        root_path, flag = self.create_folder(FileName) # can be substituded with other name
        if flag:
            # build graph
            fp = os.path.join(os.path.join(os.path.join(self.readpath, 'NTCIR'), 'data_search_e_data'), FileName)

            self.initialize()

            with open(fp, encoding = "utf-8") as xmlfile:
                # to be continue
                row_data = xmltodict.parse(xmlfile.read())
                dict_data = json.loads(json.dumps(row_data))
                self.dict_generator(dict_data)
            # need to sava graph here
            self.save(FileName)

    def handle_xlsx(self): #Todo
        return

    def handle_rdf(self, FileName):
        root_path, flag = self.create_folder(FileName) # can be substituded with other name
        if flag:
            # build graph
            fp = os.path.join(os.path.join(os.path.join(self.readpath, 'NTCIR'), 'data_search_e_data'), FileName)

            self.initialize()

            g = Graph()
            g.parse(fp, format='xml')

            for s, p, o in g:  #这里同一实体在不同三元组中Add了多次，应该加以改进
                s_label = util.split_iri(s)[1]
                p_label = util.split_iri(p)[1]
                o_label = util.split_iri(o)[1]

                #print(s_label, p_label, o_label)
                sub_id = self.add_term(s_label, 0)
                pre_id = self.add_term(p_label, 2)
                if o_label is None:
                    o_label = str(o)
                    obj_id = self.add_term(o_label, 1) # 这里的值(""与0)有待商榷
                else:
                    obj_id = self.add_term(o_label, 0)
                self.add_triple(sub_id, pre_id, obj_id)
            self.save(FileName)
    
    def run(self):
        with ProcessPool(max_workers=15) as pool:
            for line in self.data_tuple:
                Query, DatasetID, FileID, FileName, FileFormat, Qrelvs = line
                if FileFormat == 'rdf':
                    future = pool.schedule(self.handle_rdf, (FileName, ))
                elif FileFormat == 'xml':
                    future = pool.schedule(self.handle_xml, (FileName, ))
                elif FileFormat == 'json':
                    future = pool.schedule(self.handle_json, (FileName, ))
                elif FileFormat == 'csv':
                    future = pool.schedule(self.handle_csv, (FileName, ))
                else:
                    continue
                future.add_done_callback(self.task_done)

def main():
    rpath = '/path/to/data'
    wpath = '/path/to/graph'
    Worker = Builder(rpath, wpath)
    Worker.read()
    Worker.run()
    '''
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        if 'c' in arg1:
            Worker.handle_csv()
        if 'j' in arg1:
            Worker.handle_json()
        if 's' in arg1:
            Worker.handle_xls()
        if 'm' in arg1:
            Worker.handle_xml()
        if 'x' in arg1:
            Worker.handle_xlsx()
        if 'r' in arg1:
            Worker.handle_rdf()
    else:
        Worker.handle_csv()
        Worker.handle_json()
        Worker.handle_xml()
        Worker.handle_xls()
        Worker.handle_xlsx()
        Worker.handle_rdf()
    '''

if __name__ == "__main__":
    main()
    pass
