# -*- coding: utf-8 -*-
# Created on Sun Oct  4 23:18:11 2015
# Implemented in Python 3.4.0
# Author: Yun-Jhong Wu
# E-mail: yjwu@umich.edu

# Lazy GSI and his sloppy grader v2.0

import numpy as np
import re
import signal
import sys
import types
import zipfile

from copy import deepcopy
from imp import load_source
from numpy import median, allclose
from numpy.linalg import norm
from numpy.random import seed, normal, binomial, uniform, dirichlet
from os import listdir, makedirs, walk
from os.path import isdir
from shutil import copy, rmtree
from subprocess import PIPE, Popen
from timeit import timeit


def folder_setup():
    """ Create folders. """

    for folder in ['output', 'unknown_format']:
        if not isdir(folder):
            makedirs(folder)


def file_checker(folder, uniquename, topics, comments):
    """ Find and unzip zip files. Move file to test_scripts. """

    root = 'submissions/hw2/{0}'.format(folder)
    filename = 'assignment_two_{0}.zip'.format(uniquename)
    found = False
    found_invalid = False
    for path, _, files in walk(root):

        for f in files:
            if f == filename:
                pathname = "{0}/{1}".format(path, f)
                try:
                    with zipfile.ZipFile(pathname, 'r') as z:
                        z.extractall(root)
                    found = True
                    break
                except:
                    print "Can't uncompress {0}.".format(file)
                    copy(pathname, 'unknown_format')
                    comments.append('#I can\'t uncompress this zip file.')
                    found_invalid = True
        if found:
            break

    found_it = dict.fromkeys(topics, False)
    if not found and not found_invalid:
        comments.append('#I can\'t find any valid zip file.')
        return found_it

    for path, _, files in walk(root):
        for f in files:
            task = f[:-3]
            if task in topics:
                copy("{0}/{1}".format(path, f),
                     "test_scripts/{0}".format(f))
                found_it[task] = True
    return found_it


def pep8_checker(filename, pep8_exceptions=[]):
    """ Do pep8 check but ignore excpetions. """

    evaluation = Popen(["pep8", filename], stdout=PIPE).communicate()[0]
    if len(evaluation) == 0:
        return 1

    if pep8_exceptions:
        for line in evaluation.split('\n'):
            for message in pep8_exceptions:
                if len(line) and line[-len(message):] != message:
                    return 0
        return 1

    return 0


def split_file(topic, modules_list, interceptors, comments=[], sol=False):
    """ Split files by functions. """

    def module_checker(line, topic, modules, comments):
        pkgs = []
        match = re.search('from (.*) import (.*)', line)
        if match:
            funcs = ["{0}.{1}".format(match.group(1), func.strip())
                     for func in match.group(2).split(',')]
            pkgs.extend([func for func in funcs if func not in modules])
        else:
            match = re.search('import (.*) as', line)
            if match:
                if match.group(1) not in modules:
                    pkgs.append(match.group(1))
            else:
                match = re.search('import (.*)', line)
                if match:
                    pkgs.extend(map(lambda s: s.strip(),
                                    match.group(1).split(',')))

        comments.append("#Unexpected module(s) " + ','.join(pkgs))

    folder = 'solutions/hw2' if sol else 'test_scripts'
    with open("{0}/{1}.py".format(folder, topic)) as raw:
        buf = ''
        func_buf = {}
        func = None
        local_func = False
        for line in raw:
            line = line.replace('\t', '        ')
            # Local function
            if func:
                if line[0] in [' ', '\t'] and line.strip()[:4] == 'def ':
                    local_func = True
                elif len(line) > 4 and line[4] not in [' ', '\t']:
                    local_func = False

            # End of a function
            if line[:4] == 'def ' or line[:3] == 'if ':
                if func and func in interceptors[topic]:
                    setup_interceptors(func_buf, func,
                                       interceptors[topic][func])

                if line[:3] == 'if ':
                    func = None
                    break
                func = re.search('def (.*)\(', line).group(1)
                func_buf[func] = []
                return_count = 0
            elif line[0] not in ['\t', ' ', '\n']:
                continue

            # Rewrite return
            if not local_func:
                if line.strip()[:6] == 'return':
                    indent = line.split('return')[0]
                    return_count += 1
                    func_buf[func].append(indent + 'def _default():\n    ')

            if func:
                func_buf[func].append(line)

            if line[-2:] == '\\\n':
                buf += line[:-2]
                continue
            else:
                buf += line
                module_checker(buf, topic, modules_list[topic], comments)
                buf = ''

    folder_topic = '{0}/{1}'.format(folder, topic)
    if isdir(folder_topic):
        rmtree(folder_topic)
    makedirs(folder_topic)
    for func in func_buf:
        with open("{0}/{1}.py".format(folder_topic, func),
                  'w') as script:
            script.write(''.join(func_buf[func]))


def setup_interceptors(func_buf, func, interceptor):
    while func_buf[func]:
        line = func_buf[func].pop()
        text = line.strip()
        if text:
            bug = "{{{0}}}".format(','.join("'{0}':{1}".format(t, v)
                                            for t, v in interceptor))

            func_buf[func].append(line)
            func_buf[func].append("    return {0}".format(bug))

            break


def tasks_to_interceptors(tasks):
    """ Create interceptors """

    interceptors = {key: {} for key in tasks}
    test_func = {key: set() for key in tasks}
    for topic in tasks:
        for task in tasks[topic]:
            func, ttype, var_name, score = tasks[topic][task]
            interceptors[topic].setdefault(func, []).append((task, var_name))

            test_func[topic].add(func)

    return interceptors, test_func


def create_test_script(topic, func, sol=False):
    """ Override a function in the solution. """

    folder = 'solutions/hw2' if sol else 'test_scripts'
    filename = "{0}/test_{1}_{2}.py".format(folder, topic, func)
    copy("solutions/hw2/{0}.py".format(topic), filename)
    with open(filename, 'a') as sol, \
            open("{0}/{1}/{2}.py".format(folder, topic, func)) as func:
        for line in func:
            sol.write(line)


def timeout(limit):
    def wrap(func):
        """ Decorator: Function call has to be done within time limit. """

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


def timer(filename, uniquename, task, n_rep=1, limit=60):
    """ Get median of running time of main(). """

    @timeout(limit)
    def subtimer(script):
        try:
            return timeit(script, number=1,
                          setup='from subprocess import call; import os')
        except:
            return float('inf')

    script = """with open(os.devnull, 'w') as DEVNULL:
                call(['python', '{0}'], 
                stdout=DEVNULL)""".format(filename)

    return median([subtimer(script) for _ in range(n_rep)])


def sloppy_grader(uniquename, topics, tasks, modules_list,
                  interceptors, test_funcs, test_cases,
                  evaluation, comments, timeout, submissions):

    def equal(a, b, local_func_par=uniform(size=5)):
        if isinstance(a, types.FunctionType):
            if isinstance(b, types.FunctionType):
                try:
                    a(local_func_par)
                except:
                    local_func_par = uniform(size=50)
                try:
                    b_output = b(local_func_par)
                except:
                    return False
                return equal(a(local_func_par),
                             b_output)
            else:
                return False
        else:
            if isinstance(a, np.ndarray):
                a = np.squeeze(np.array(a))
                b = np.squeeze(np.array(b))
                if len(a.shape) == 1:
                    a = np.sort(a)
                if len(b.shape) == 1:
                    b = np.sort(b)
                return allclose(a, b, rtol=1e-4) if a.shape == b.shape else False
            elif hasattr(a, '__iter__'):
                return all([equal(i, j) for i, j in zip(a, b)])
            else:
                return a == b

    @timeout
    def var_checker(solution, submission, func, args, tasks, comments):
        args1 = deepcopy(args)
        args2 = deepcopy(args)

        seed(53)
        sol = solution(*args1)
        func_score = 0
        seed(53)

        try:
            sub = submission(*args2)
        except:
            sub = dict.fromkeys(sol.keys(), 0)

        for task in tasks:
            if tasks[task][0] == func:
                passed = equal(sol[task], sub[task])
                if hasattr(passed, '__iter__'):
                    passed = passed[0]
                score = tasks[task][-1] * passed
                comments.append("{0} +{1}".format(task, score))
                func_score += score

        return func_score

    full_topic_scores = dict(zip(topics, [7, 7, 6]))
    score = 0
    all_passed = dict.fromkeys(topics, False)
    for topic in topics:
        if submissions[topic]:
            split_file(topic, modules_list, interceptors)
            topic_score = 0
            for func in test_funcs[topic]:
                filename = "test_{0}_{1}.py".format(topic, func)
                sol = load_source('sol', "solutions/hw2/{0}".format(filename))
                create_test_script(topic, func)
                try:
                    sub = load_source(
                        'sub', "test_scripts/{0}".format(filename))
                except SyntaxError:
                    comments.append("#Syntax error in {0}.{1}".format(topic,
                                                                      func))
                except:
                    comments.append("#Unknown error in {0}.{1}".format(topic,
                                                                       func))

                sol_func = getattr(sol, func)
                sub_func = getattr(sub, func)

                topic_score += var_checker(sol_func, sub_func, func,
                                           test_cases[topic][func],
                                           tasks[topic], comments)
            score += topic_score
            all_passed[topic] = topic_score == full_topic_scores[topic]
        else:
            comments.append("{0}.py not found".format(topic))

    return score, all_passed


def lazy_GSI(topics, suppl, tasks, modules_list):
    """ main function """

    # Preparing folders and files
    folder_setup()
    test_cases = test_case_generator()
    interceptors, test_funcs = tasks_to_interceptors(tasks)
    grades = []

    for s in suppl:
        copy("suppl/hw2/{0}".format(s), './')

    for topic in topics:
        split_file(topic, modules_list, interceptors, sol=True)
        for func in test_funcs[topic]:
            create_test_script(topic, func, sol=True)

    for folder in listdir('submissions/hw2'):
        # Get uniquenames
        match = re.search('(.*), (.*)\((.*)\)', folder)
        if match:
            uniquename = match.group(3)
            name = "{0} {1}".format(match.group(2), match.group(1))
        else:
            continue

        print "{1} Grading: {0} {1}".format(folder, '=' * 30)
        if isdir('test_scripts'):
            rmtree('test_scripts')
        makedirs('test_scripts')

        evaluation = [uniquename, name, 0]
        comments = []

        submissions = file_checker(folder, uniquename,
                                   topics, comments)
        for s in suppl:
            copy("suppl/hw2/{0}".format(s), 'test_scripts')

        # PEP8 check
        for s in submissions:
            filename = "{0}.py".format(s)
            pep8_pass = pep8_checker(filename, [])
            if pep8_pass:
                evaluation[2] += 1
            comments.append("{0} pep8_check +{1}".format(filename.split('_')[2][:-3],
                                                         pep8_pass))

        score, all_passed = sloppy_grader(uniquename, topics,
                                          tasks, modules_list,
                                          interceptors, test_funcs,
                                          test_cases, evaluation,
                                          comments, timeout(30),
                                          submissions)
        evaluation[2] += score

        if all_passed['assignment_two_adaboost']:
            no_loop(evaluation, comments)

        # Running time
        running_time = []
        for i, topic in enumerate(topics):
            filename = "test_scripts/{0}.py".format(topic)
            if all_passed[topic]:
                try:
                    running_time.append(timer(filename, uniquename,
                                              topic, 1))
                    state = 'Evaluated'
                except:
                    running_time.append(float('inf'))
                    state = 'Failed'
            else:
                running_time.append(float('inf'))
                state = 'Skipped'
            print "Running time: {0}, {1}: {2}".format(uniquename, topic, state)
        evaluation.extend(running_time)
        evaluation.extend(sorted(comments))
        grades.append(evaluation)

    # Remove files
    if isdir('test_scripts'):
        rmtree("test_scripts")
    with open('output/evaluation.csv', 'w') as output:
        for student in grades:
            output.write(','.join(map(str, student)))
            output.write('\n')


def test_case_generator():
    sys.path.append('solutions/hw2')
    sys.path.append('suppl/hw2')
    from assignment_two_adaboost import weak_learner as wl
    sys.path.pop()
    sys.path.pop()
    from kernels import rbf

    seed(1)
    instances = normal(size=(50, 5))
    labels = binomial(1, 0.5, 50)
    dist = dirichlet(uniform(size=50))
    ker = rbf(1)
    mat = uniform(size=(5, 5))
    mat = (mat / np.sum(mat, axis=1)).T
    test_cases = {'assignment_two_adaboost': {
        'compute_error':
        [lambda x: x[3] < 0.2, instances, labels, dist],
        'run_adaboost':
        [instances, labels, wl],
        'update_dist':
        [lambda x: x[2] > -0.2, instances,
         labels, dist, normal()],
        'weak_learner': [instances, labels, dist]},
        'assignment_two_pagerank': {'compute_pageranks': [mat],
                                    'main': []},
        'assignment_two_svm': {
        'evaluate_classifier':
        [lambda x: norm(x) > 5, instances, labels],
        'svm_train': [instances, labels, ker]}}

    return test_cases


def no_loop(evaluation, comments):
    with open('test_scripts/assignment_two_adaboost.py', 'r') as data:
        contents = data.read()
        count = contents.count('for')
        count += contents.count('while')
    if count < 3:
        evaluation[2] += 2
        comments.append("Extra credit for no additional loops in adaboost: +2")

if __name__ == '__main__':
    topics = ['assignment_two_pagerank',
              'assignment_two_svm',
              'assignment_two_adaboost']

    suppl = ['example_index.txt',
             'example_arcs.txt',
             'ionosphere.data',
             'ionosphere.data.txt',
             'kernels.py']

    tasks = {'assignment_two_pagerank':
             {'TASK1.1': ('main', 'local', 'adj_mat', 0.5),
              'TASK1.2': ('main', 'local', 'dangling', 0.5),
              'TASK1.3': ('main', 'local', 'adj_mat_norm', 0.5),
              'TASK1.4': ('main', 'local', 'adj_mat_stoch', 0.5),
              'TASK1.5': ('compute_pageranks', 'output', '_default()', 2),
              'TASK1.6': ('main', 'local', 'top_10', 1),
              'TASK1.7': ('main', 'local', 'close_to_1', 1),
              'TASK1.8': ('main', 'local', 'pageranks_eig', 1)},
             'assignment_two_svm':
             {'TASK2.1': ('svm_train', 'local', 'kernel_mat', 0.5),
              'TASK2.2': ('svm_train', 'local_func', 'func', 0.5),
              'TASK2.3': ('svm_train', 'local_func', 'func_deriv', 1),
              'TASK2.4': ('svm_train', 'local', 'box_constraints', 0.5),
              'TASK2.5': ('svm_train', 'local', 'alpha_y_nz', 1),
              'TASK2.6': ('svm_train', 'local', 'support_vectors', 0.5),
              'TASK2.7': ('svm_train', 'output', '_default()', 1),
              'TASK2.8': ('evaluate_classifier', 'local',
                          '[pos_labels,pos_correct,neg_labels,neg_correct]', 2)},
             'assignment_two_adaboost':
             {'TASK3.1': ('weak_learner', 'output', '_default()', 2),
              'TASK3.2': ('compute_error', 'output', '_default()', 1),
              'TASK3.3': ('update_dist', 'output', '_default()', 2),
              'TASK3.4': ('run_adaboost', 'output', '_default()', 1)}
             }

    modules_list = {'assignment_two_pagerank':
                    set(['numpy', 'numpy.linalg']),
                    'assignment_two_svm':
                        set(['numpy', 'kernels',
                             'scipy.optimize.fmin_l_bfgs_b']),
                    'assignment_two_adaboost':
                        set(['math', 'numpy', 'assignment_two_svm'])}

    lazy_GSI(topics, suppl, tasks, modules_list)
