# -*- coding: utf-8 -*-
# Created on Wed Sep 30 15:50:02 2015
# Implemented in Python 2.7.10
# Author: Yun-Jhong Wu
# E-mail: yjwu@umich.edu

# Lazy GSI and his sloppy grader

from numpy import median
from os import listdir, remove, walk
from shutil import copy
from subprocess import Popen, PIPE
from timeit import timeit
import random
import re

from sloppy_grader import sloppy_grader


def file_finder(pathname, comments):

    for p, _, files in walk(pathname):
        for file in files:
            if file[:-3] in tasks:
                try:
                    copy("{0}/{1}".format(p, file),
                         "test_scripts/{0}".format(file))
                    with open("test_scripts/{0}".format(file)) as script:
                        for line in script:
                            for package in banned:
                                if package in line:
                                    comments.append('banned')
                except:
                    comments.append('not_found')


def timer(filename, uniquename, task, n_rep):

    script = """with open(os.devnull, 'w') as DEVNULL:
                call(['python', '{0}'],
                     stdout=DEVNULL)""".format(filename)
    try:
        t = [timeit(script, number=1,
                    setup='from subprocess import call; import os')
             for _ in range(n_rep)]
        print("Time {0}, {1}: Evaluated".format(uniquename, task))
        return median(t)
    except:
        print("Time {0}, {1}: Failed".format(uniquename, task))
        return float('inf')


def pep8_report(report, exceptions):

    for line in report.split('\n'):
        for message in exceptions:
            if len(line) and line[-len(message):] != message:
                return 0
    return 1


def line_counter(path, sol=False):

    lines = {}
    for i, task in enumerate(tasks):
        filename = '{0}/{1}{2}.py'.format(path, task, '_sol' if sol else '')
        for method in scores[task]:
            lines[method] = 0
        method = ''
        with open(filename, 'r') as script:
            for line in script:
                match = re.search('def (.*)\(', line)
                if match:
                    method = match.group(1)
                if method in scores[task]:
                    lines[method] += line.lstrip(' ')[0] not in ['\n', '#']

    return lines

if __name__ == '__main__':
    time_check = True
    pep8_check = True

    n_rep = 5

    tasks = ['assignment_one_kmeans',
             'assignment_one_optimization',
             'assignment_one_nearest_neighbor']

    suppl = ['seeds_dataset.txt',
             'seeds_dataset_shuffled.txt',
             'losses.py']

    scores = {'assignment_one_kmeans':
              {'read_data': 1,
               'num_unique_labels': 1,
               'kmeans_plus_plus': 2,
               'assign_cluster_ids': 1,
               'recompute_centers': 1},
              'assignment_one_optimization':
              {'gradient_descent': 2,
               'coordinate_descent': 2,
               'loss_grad_calculator': 1,
               'loss_grad_1d_calculator': 1},
              'assignment_one_nearest_neighbor':
              {'get_fold_indices': 2,
               'nn_classifier': 2,
               'classification_error': 1}}

    pep8_exceptions = ["do not assign a lambda expression, use a def"]

    methods = [method for task in scores for method in scores[task]]
    banned = ['numpy', 'scipy', 'sklearn', 'cvxopt']
    line_sol = line_counter('hw1_sol', True)

    performance = [('uniquename', 'time_1, time_2, time_3',
                    'pep8_1, pep8_2, pep8_3',
                    ', '.join(methods), 'comments')]
    seed = 623
    for s in suppl:
        copy("suppl/{0}".format(s), './')
        copy("suppl/{0}".format(s), './test_scripts/')

    for folder in listdir("hw1"):
        _, _, uniquename = folder.split('_')
        print("{1} Grading: {0} {1}".format(uniquename, '=' * 30))
        pathname = "hw1/assignment_one_{0}/".format(uniquename)
        comments = []

        # Find scripts and move them to ./test_scripts/, check modules
        file_finder(pathname, comments)

        # Check #lines
        for method, k in line_counter('test_scripts').items():
            if abs(line_sol[method] - k) > 2:
                comments.append('#line_{0}'.format(method))

        # Variables for storing results
        running_time = []
        pep8_passed = []
        results = sloppy_grader(scores)

        random.seed(seed)
        for i, task in enumerate(tasks):
            filename = "test_scripts/{0}.py".format(task)
            # Running time
            if time_check and all(results[m] for m in scores[task]):
                running_time.append(timer(filename, uniquename, task, n_rep))
            else:
                running_time.append(float('inf'))
                print("Time {0}, {1}: Skipped".format(uniquename, task))
            # pep8 check
            if pep8_check:
                pep8_passed.append(pep8_report(Popen(["pep8", filename],
                                               stdout=PIPE).communicate()[0],
                                               pep8_exceptions))

        # Cleanups
        for file in tasks:
            remove("test_scripts/{0}.py".format(file))
            remove("test_scripts/{0}.pyc".format(file))

        performance.append(', '.join((uniquename,
                                      ', '.join(map(str, running_time)),
                                      ', '.join(map(str, pep8_passed)),
                                      ', '.join(map(str, [results[m]
                                                          for m in methods])),
                                      ';'.join(comments))))
        print performance[-1]

    # Final cleanups
    for s in suppl:
        remove("{0}".format(s))
    for s in listdir('test_scripts'):
        remove("test_scripts/{0}".format(s))

    with open("evaluation.csv", "w") as output:
        for student in performance:
            output.write(student)
            output.write('\n')
