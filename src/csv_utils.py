'''
Created on Nov 29, 2015

@author: Yohan
'''
import csv 
import sys

csv.field_size_limit(sys.maxsize)

def iter_csv(path, delimiter=",", header_exists=True, header=None, map_format=True, **kwargs):
    in_file = open(path)
    if header_exists:
        in_csv = csv.reader(in_file, delimiter=delimiter, **kwargs)
        if header:
            for h in in_csv.next(): header.append(h)
        else:
            header = in_csv.next()
    if map_format:
        in_csv = csv.DictReader(in_file, delimiter=delimiter, fieldnames=header, **kwargs)
    elif not header_exists: in_csv = csv.reader(in_file, delimiter=delimiter, **kwargs)
    
    for row in in_csv:
        yield row
    in_file.close()
    
def iter_csv_noheader(path, delimiter=",", readmode='r', encoding="utf8", **kwargs):
    in_file = open(path, readmode, encoding=encoding)
    in_csv = csv.reader(in_file, delimiter=delimiter, **kwargs)
    for row in in_csv:
        yield row
    in_file.close()
    
def iter_csv_header(path, delimiter=",", header=None, map_format=True, readmode='r', **kwargs):
    in_file = open(path,readmode)
    in_csv = csv.reader(in_file, delimiter=delimiter, **kwargs)
    tmp_header = in_csv.__next__()
    if header != None:
        for h in tmp_header: header.append(h)
    if map_format: 
        in_csv = csv.DictReader(in_file, delimiter=delimiter, fieldnames=tmp_header, **kwargs)
    else: 
        in_csv = csv.reader(in_file, delimiter=delimiter, **kwargs)
    
    for row in in_csv:
        yield row
    in_file.close()
    
    
def print_dict_csv(dic, path, delimiter=","):
    out_file = open(path,"w")
    out_csv = csv.writer(out_file, delimiter=delimiter)
    for k,v in dic.iteritems():
        out_csv.writerow([k,v])
    out_file.close()
