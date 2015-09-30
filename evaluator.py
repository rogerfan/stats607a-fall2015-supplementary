# -*- coding: utf-8 -*-
# Created on Fri Sep 25 17:55:10 2015
# Implemented in Python 3.4.0
# Author: Yun-Jhong Wu
# E-mail: yjwu@umich.edu

from os import chdir, listdir, remove, walk
from shutil import copyfile
from timeit import timeit
from numpy import median
from subprocess import Popen, PIPE
from sys import path

import random

#from task_checker import task_checker

if __name__ == '__main__':
    path.append("../../modules")
    filenames = ['assignment_one_kmeans.py',
                 'assignment_one_nearest_neighbor.py',
                 'assignment_one_optimization.py']
    suppl = ['seeds_dataset.txt', 'seeds_dataset_shuffled.txt', 'losses.py']
    banned = ['numpy', 'scipy', 'sklearn', 'cvxopt']
    
    performance = []
    seed = 623
    for folder in listdir("hw1"):
        _, _, uniquename = folder.split('_')
        pathname = "hw1/assignment_one_{0}/".format(uniquename)
        comment = []
        # Find scripts and move them to assignment_one_{uniquename}
        for p, _, files in walk(pathname):
            for file in files:
                if file in filenames:
                    try:
                        copyfile('{0}/{1}'.format(p, file), pathname + file)
                    except:
                        pass
                    
                    with open('{0}/{1}'.format(pathname, file)) as script:
                        for line in script:
                            for package in banned:
                                if package in line:
                                    comment.append('banned')
                                
                    
        for s in suppl: copyfile("suppl/{0}".format(s), pathname + s)

        # Variables for storing results        
        running_time = []        
        results = [] #task_checker(pathname)
        
        chdir(pathname)        
        random.seed(seed)
        if all(results):
            for i, task in enumerate(filenames):
                script = "call(['python', '{0}'])".format(task)
                try: 
                    t = [timeit(script, number=1, 
                                setup='from subprocess import call') 
                                for _ in range(5)]
                    running_time.append(median(t))
                    print("{0}, {1}: Evaluated".format(uniquename, task))
                except:
                    running_time.append(float('inf'))
                    print("{0}, {1}: Failed".format(uniquename, task))
            
        pep8_failed = [len(Popen(["pep8", task], stdout=PIPE).communicate()[0])
                       for task in filenames]
        for s in suppl: remove(s)
        chdir("../..")
        
        performance.append((uniquename, 
                            ' '.join(map(str, running_time)), 
                            ' '.join(map(str, pep8_failed)), 
                            ' '.join(map(str, results)),
                            ' '.join(comment)))

    with open("evaluation.txt", "w") as output:
        for student in performance:
            output.write(student)
            output.write('\n')