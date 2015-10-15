# -*- coding: utf-8 -*-
# Created on Wed Sep 30 15:50:02 2015
# Implemented in Python 2.7.10
# Author: Yun-Jhong Wu
# E-mail: yjwu@umich.edu

# Lazy GSI and his sloppy grader

# hw?: assignments downloaded from ctools
# hw?_sol: solutions
# suppl: datasets, modules
# unknown_format: found compressed file but can't uncompressed it
# test_sciprts: scripts to be graded
# output

from copy import deepcopy
from numpy import median
from os import listdir, makedirs, remove, walk
from os.path import isdir, isfile
from shutil import copy
from subprocess import Popen, PIPE
from timeit import timeit
import signal
import random
import re
import zipfile

from sloppy_grader import sloppy_grader


def file_checker(root, filename, comments, tasks, modules):
    """ Find scripts and move them to ./test_scripts/; check modules. """

    def module_check(comments, script, modules):
        buffer = ''
        pkgs = []
        for line in script:
            if line[-2:] == '\\\n':
                buffer += line[:-2]
                continue
            else:
                buffer += line
                match = re.search('from (.*) import', buffer)
                if match:
                    if match.group(1) not in modules:
                        pkgs.append(match.group(1))
                else:
                    match = re.search('import (.*)', buffer)
                    if match:
                        pkgs.extend(map(lambda s: s.strip(),
                                    match.group(1).split(',')))
                buffer = ''

        comments.extend(["#module_{0}".format(pkg) for pkg in pkgs
                         if pkg not in modules])

    found = False
    found_invalid = False
    for path, _, files in walk(root):
        for file in files:
            if file == filename:
                pathname = "{0}/{1}".format(path, file)
                try:
                    with zipfile.ZipFile(pathname, 'r') as z:
                        z.extractall(root)
                    found = True
                    break
                except:
                    print "Can't uncompress {0}.".format(file)
                    copy(pathname, 'unknown_format')
                    comments.append('#failed_to_uncompress')
                    found_invalid = True
        if found:
            break
    if not found and not found_invalid:
        comments.append('#zip_not_found')
        

    found_it = dict.fromkeys(tasks, False)
    for path, _, files in walk(root):
        for file in files:
            task = file[:-3]
            if task in tasks:
                
                found_it[task] = True
                copy("{0}/{1}".format(path, file),
                     "test_scripts/{0}".format(file))
                with open("test_scripts/{0}".format(file)) as script:
                    module_check(comments, script, modules[task])
    return found_it

def timeout(limit):
    def wrap(func):
        """ Decorator: Each function call has to be done within time limit. """
    
        def _handler(signum, frame):
            print 'Timeout'
            raise Exception('Timeout')
    
        def func_with_timeout(*args, **kwargs):
            handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(limit)
            output = func(*args, **kwargs)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(0)
            return output
    
        func_with_timeout.__name__ = func.__name__
        return func_with_timeout
    return wrap

def line_counter(path, found_it, sol=False):
    """ There must be something wrong if you wrote terribly long codes. """

    def counter(script, nlines):
        in_string = False
        method = None
        for line in script:
            if line[:4] == 'def ':
                match = re.search('def (.*)\(', line)
                if match and match.group(1) in scores[task]:
                    if match.group(1) in scores[task]:
                        method = match.group(1)
                        nlines[method] = 0
                    else:
                        method = None
                elif line[0] not in ['\s', '\t']:
                    method = None

            if method:
                # Remove Documents
                s = len(re.findall('"""', line))
                if line.lstrip()[:3] == '"""':
                    if s % 2:
                        in_string = not in_string
                if in_string:
                    in_string = len(re.findall('"""', line)) % 2 == 1

                # Remove blank lines and comments
                if line.lstrip(' ')[0] not in ['\n', '#']:
                    if not in_string:
                        nlines[method] += 1

    nlines = {}
    for i, task in enumerate(tasks):
        filename = '{0}/{1}{2}.py'.format(path, task, '_sol' if sol else '')
        for method in scores[task]:
            nlines[method] = 0
        method = None
        if found_it[task]:
            with open(filename, 'r') as script:
                counter(script, nlines)
        else:
            for method in scores[task]:
                nlines[method] = float('inf')
    return nlines

def eavesdropper(path, found_it, sol=False):
    """ There must be something wrong if you wrote terribly long codes. """

    def setup_intercepts(script, targets):
        method = None
        for line in script:
            if line[:4] == 'def ':
                match = re.search('def (.*)\(', line)
                if match and match.group(1) in scores[task]:
                    method = (match.group(1) if match.group(1) in targets[task]
                              else None)
                elif line[0] not in ['\s', '\t']:
                    method = None

            if method:
                # Remove Documents
                s = len(re.findall('"""', line))
                if line.lstrip()[:3] == '"""':
                    if s % 2:
                        in_string = not in_string
                if in_string:
                    in_string = len(re.findall('"""', line)) % 2 == 1

                # Remove blank lines and comments
                if not in_string and line[:11] == '    return ':
                    line += ', ' + targets[method]

    nlines = {}
    for i, task in enumerate(tasks):
        filename = '{0}/{1}{2}.py'.format(path, task, '_sol' if sol else '')
        for method in scores[task]:
            nlines[method] = 0
        method = None
        if found_it[task]:
            with open(filename, 'r') as script:
                counter(script, nlines)
        else:
            for method in scores[task]:
                nlines[method] = float('inf')
    return nlines


def pep8_report(report, exceptions):
    """ Do pep8 check but ignore excpetions. """

    for line in report.split('\n'):
        for message in exceptions:
            if len(line) and line[-len(message):] != message:
                return 0
    return 1


def timer(filename, uniquename, task, n_rep):
    """ Get median of running time of main(). """

    @timeout(300)
    def subtimer(script):
        try:
            return timeit(script, number=1,
                          setup='from subprocess import call; import os')
        except:
            return float('inf')

    script = """with open(os.devnull, 'w') as DEVNULL:
                call(['python', '{0}'], stdout=DEVNULL)""".format(filename)
    return median([subtimer(script) for _ in range(n_rep)])


if __name__ == '__main__':
    time_check = True
    n_rep = 1
    seed = 623

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
    modules = {tasks[0]: set(['random', 'time']),
               tasks[1]: set(['math', 'random', 'time', 'losses']),
               tasks[2]: set(['math', 'random', 'assignment_one_kmeans'])}
    line_sol = line_counter('hw1_sol', dict.fromkeys(tasks, True), True)
    records = [','.join(('uniquename', 'name', 'time_1,time_2,time_3',
                         'pep8_1,pep8_2,pep8_3',
                         ','.join(methods), 'comments'))]
    evaluated = set()
    for folder in ['output', 'unknown_format', 'test_scripts']:
        if not isdir(folder):
            makedirs(folder)

    # Skip assignments who have got graded
    try:
        with open('output/evaluation.csv') as students:
            students.readline()
            for student in students:
                evaluated.add(student.split(',')[0])
    except:
        pass

    for s in suppl:
        copy("suppl/hw1/{0}".format(s), './')

    for folder in listdir('hw1'):
        match = re.search('(.*), (.*)\((.*)\)', folder)
        if match:
            uniquename = match.group(3)
            name = "{0} {1}".format(match.group(2), match.group(1))
        else:
            continue

        if uniquename in evaluated:
            print "Found {0} in evaluation.csv".format(folder)
            continue

        print "{1} Grading: {0} {1}".format(folder, '=' * 30)
        comments = []

        # Prepare files
        for s in suppl:
            copy("suppl/hw1/{0}".format(s), './test_scripts/')

        filename = 'assignment_one_{0}.zip'.format(uniquename)
        root = 'hw1/{0}'.format(folder)
        found_it = file_checker(root, filename, comments, tasks, modules)

        # Check #lines
        for method, k in line_counter('test_scripts', found_it).items():
            if float('inf') > abs(line_sol[method] - k) > 3:
                comments.append('#line_{0}'.format(method))

        # Variables for storing results
        running_time = []
        pep8_passed = []

        # Sloppy grader is grading your homework
        # but doesn't want you know what's going on.
        results = sloppy_grader(scores, found_it, comments, timeout(30))

        random.seed(seed)
        for i, task in enumerate(tasks):
            filename = "test_scripts/{0}.py".format(task)
            # Running time
            if (time_check and found_it[task] and
               all(results[m] for m in scores[task])):
                try:
                    running_time.append(timer(filename, uniquename,
                                              task, n_rep))
                    state = 'Evaluated'
                except:
                    running_time.append(float('inf'))
                    state = 'Failed'
            else:
                running_time.append(float('inf'))
                state = 'Skipped'
            print "Time {0}, {1}: {2}".format(uniquename, task, state)

            # pep8 check
            pep8_passed.append(pep8_report(Popen(["pep8", filename],
                                           stdout=PIPE).communicate()[0],
                                           pep8_exceptions))

        # Cleanups
        for file in listdir('test_scripts'):
            remove("test_scripts/{0}".format(file))

        records.append(','.join((uniquename, name,
                                 ','.join(map(str, running_time)),
                                 ','.join(map(str, pep8_passed)),
                                 ','.join(map(str, [results[m]
                                                    for m in methods])),
                                 ';'.join(comments))))
        print records[-1]

    # Final cleanups
    for s in suppl:
        remove("{0}".format(s))
    existing_report = isfile('output/evaluation.csv')
    with open("output/evaluation.csv",
              'a' if existing_report else 'w') as output:
        for student in records:
            output.write(student)
            output.write('\n')
