# -*- coding: utf-8 -*-
# Created on Fri Sep 25 17:55:10 2015
# Implemented in Python 3.4.0
# Author: Yun-Jhong Wu
# E-mail: yjwu@umich.edu

from os import chdir, listdir, remove
from shutil import copyfile
from timeit import timeit
from numpy import median
from subprocess import Popen, PIPE
from sys import path
import random

if __name__ == '__main__':
    path.append("../../modules")
    filenames = ['assignment_one_kmeans.py',
                 'assignment_one_nearest_neighbor.py',
                 'assignment_one_optimization.py']
    performance = []
    for folder in listdir("hw"):
        seed = random.randint(0, 65535)
        _, _, uniquename = folder.split('_')
        pathname = "hw/assignment_one_{0}/".format(uniquename)
        dataname = "seeds_dataset.txt"
        copyfile("datasets/{0}".format(dataname), pathname + dataname)
        t = [float('inf')] * 3
        pep8_pass = True
        chdir(pathname)
        for i, task in enumerate(filenames):
            random.seed(seed)
            try:
                script = "exec(open('{0}').read())".format(task)
                t[i] = median([timeit(script, number=1) for _ in range(10)])
                print("{0}, {1}: Evaluated".format(uniquename, task))
            except:
                print("{0}, {1}: Failed".format(uniquename, task))

            pep8_result = Popen(["pep8", task], stdout=PIPE).communicate()[0]
            pep8_pass &= len(pep8_result) == 0

        remove(dataname)
        chdir("../..")
        performance.append((sum(t), t[0], t[1], t[2], pep8_pass, uniquename))

    performance.sort()
    with open("evaluation.txt", "w") as output:
        for student in performance:
            output.write("{0} {1}\n".format(student[-1],
                                            ' '.join(map(str, student[:-1]))))
