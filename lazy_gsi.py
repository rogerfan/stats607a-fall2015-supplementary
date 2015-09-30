# -*- coding: utf-8 -*-
# Created on Wed Sep 30 15:50:02 2015
# Implemented in Python 2.7.10
# Author: Yun-Jhong Wu
# E-mail: yjwu@umich.edu

# Lazy GSI and his sloppy grader

from os import chdir, listdir, remove, walk
from shutil import copy
from timeit import timeit
from numpy import median
from subprocess import Popen, PIPE

import random

from sloppy_grader import sloppy_grader


def pep8_report(report):
    exceptions = ["do not assign a lambda expression, use a def"]
    for line in report.split('\n'):
        for message in exceptions:
            if len(line) and line[-len(message):] != message:
                print(line)
                return 0
    return 1

if __name__ == '__main__':
    task_check = True
    time_check = False
    pep8_check = True

    n_rep = 1

    tasks = ['assignment_one_kmeans',
             'assignment_one_optimization',
             'assignment_one_nearest_neighbor']
    suppl = ['seeds_dataset.txt',
             'seeds_dataset_shuffled.txt',
             'losses.py']
    banned = ['numpy', 'scipy', 'sklearn', 'cvxopt']

    performance = []
    seed = 623

    for s in suppl:
        copy("suppl/{0}".format(s), 'test_scripts/')
        copy("suppl/{0}".format(s), 'hw1_sol/')

    for folder in listdir("hw1"):
        _, _, uniquename = folder.split('_')
        pathname = "hw1/assignment_one_{0}/".format(uniquename)
        comment = []

        # Find scripts and move them to ./test_scripts/
        for p, _, files in walk(pathname):
            for file in files:
                if file[:-3] in tasks:
                    try:
                        copy('{0}/{1}'.format(p, file),
                             'test_scripts/{0}'.format(file))

                        with open('test_scripts/{0}'.format(file)) as script:
                            for line in script:
                                for package in banned:
                                    if package in line:
                                        comment.append('banned')
                    except:
                        comment.append('not_found')

        # Variables for storing results
        running_time = []
        pep8_passed = []
        chdir('test_scripts')
        results = sloppy_grader() if task_check else [[], [], []]
        random.seed(seed)
        for i, task in enumerate(tasks):

            if time_check and False not in results[i]:
                script = "call(['python', '{0}.py'])".format(task)
                try:
                    t = [timeit(script, number=1,
                                setup='from subprocess import call')
                         for _ in range(n_rep)]
                    running_time.append(median(t))
                    print("{0}, {1}: Evaluated".format(uniquename, task))
                except:
                    running_time.append(float('inf'))
                    print("{0}, {1}: Failed".format(uniquename, task))
            else:
                running_time.append(float('inf'))
                print("{0}, {1}: Skipped".format(uniquename, task))

            if pep8_check:
                pep8_passed.append(pep8_report(Popen(["pep8",
                                                      "{0}.py".format(task)],
                                               stdout=PIPE).communicate()[0]))
        for file in listdir('./'):
            if file[-3:] in ['.py', 'pyc']:
                remove(file)

        chdir("..")

        performance.append((uniquename,
                            ' '.join(map(str, running_time)),
                            ' '.join(map(str, pep8_passed)),
                            ' '.join(map(str, results)),
                            ';'.join(comment)))

    with open("evaluation.txt", "w") as output:
        for student in performance:
            output.write(' '.join(student))
            output.write('\n')