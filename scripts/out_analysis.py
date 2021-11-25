import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import stats


def save_to_csv(words, save_loc):
    cols = ['word', 'dominance', 'stress', 'category', 'data_seed', 'net_seed', 'hidden_size', 'weight_decay', 'ex_type', 'diff_thresh0.5', 'rt_thresh0.5',
            'st_thresh0.5', 'acc_thresh0.5', 'avg_out_diff_tick_1']
    for key in words.keys():
        row = [key]
        for i in range(1, len(cols) - 1):
            row.append(words[key][cols[i]])

        row.append('NaN')  # avg_out_diff_tick_1
        for i in range(1, 20):
            row.append(words[key]['avg_out_diff_tick_' + str(i+1)])
        for i in range(0, 20):
            row.append(words[key]['avg_out_tick_' + str(i+1)])
        words[key] = row

    out_diff_ticks = []
    for i in range(1, 20):
        out_diff_ticks.append('avg_out_diff_tick_' + str(i+1))

    out_ticks = []
    for i in range(0, 20):
        out_ticks.append('avg_out_tick_' + str(i+1))

    cols = cols + out_diff_ticks + out_ticks

    df = pd.DataFrame.from_dict(
        words,  columns=cols, orient='index')
    df.to_csv(save_loc, index=False)
    return df


def get_param_val(name):
    # input like: name-val, returns val
    val = name.split('-')[-1]
    return val


def parse_line(line, in_format, file_type):
    if in_format == 1:  # akathian project format
        entries = line.split(' ')
        out_start = 5

        name_details, epoch, net_params, tick, line_type = entries[:out_start]
        richness = 'NaN'
        if file_type == 'train':
            num, name, dominance, category, richness, data_seed, ex_type = name_details.split(
                '_')
            dominance = get_param_val(dominance)
            category = get_param_val(category)
            richness = get_param_val(richness)
        else:
            num, name, data_seed, ex_type = name_details.split('_')
            dominance = 'NaN'
            category = 'NaN'
        _, hidden_size, weight_decay, net_seed = net_params.split('_')
        hidden_size = get_param_val(hidden_size)
        weight_decay = get_param_val(weight_decay)
        net_seed = get_param_val(net_seed)
        data_seed = get_param_val(data_seed)
        tick = int(tick)
        return entries, out_start, epoch, tick, line_type, name, dominance, category, richness, hidden_size, weight_decay, net_seed, data_seed, ex_type
    elif in_format == 2:  # Ian project formet
        entries = line.split('|')
        output = entries[-1].split(' ')
        out_start = 1
        epoch, name = entries[:2]
        line_type = output[0]
    return entries, out_start, epoch, line_type, name


def stress(out):
    stress = 0
    for u in out:
        stress += u * math.log(u, 2) + (1 - u) * \
            math.log(1-u, 2) - math.log(0.5, 2)
    stress = stress / len(out)
    return stress


def prev_word_cleanup(words, prev_name):
    del words[prev_name]['out']
    if 'targ' in words[prev_name].keys():
        del words[prev_name]['targ']
    if 'rt_thresh0.5' not in words[prev_name].keys():
        words[prev_name]['rt_thresh0.5'] = 'NaN'
    if 'st_thresh0.5' not in words[prev_name].keys():
        words[prev_name]['st_thresh0.5'] = 'NaN'
    if 'acc_thresh0.5' not in words[prev_name].keys():
        words[prev_name]['acc_thresh0.5'] = 0
    if 'diff_thresh0.5' not in words[prev_name].keys():
        words[prev_name]['diff_thresh0.5'] = 'NaN'

    return words


def analyze(file, method, measure, in_format, st_threshold, file_type, save_loc):
    f = open(file, 'r')
    lines = f.readlines()
    count = 0
    words = dict()
    prev_name = ''
    out_at_tick = dict()
    for line in lines:
        if in_format == 1:
            entries, out_start, epoch, tick, line_type, name, dominance, category, richness, hidden_size, weight_decay, net_seed, data_seed, ex_type = parse_line(
                line, in_format, file_type)
        elif in_format == 2:
            entries, out_start, epoch, line_type, name = parse_line(
                line, in_format, file_type)

        if count == 0:
            prev_name = name

        if name not in out_at_tick.keys():
            out_at_tick[name] = dict()

        # next word is presented
        if prev_name != name:
            words = prev_word_cleanup(words, prev_name)
            if in_format == 2:
                tick = 0

        if name not in words.keys():
            words[name] = dict()

        words[name]['richness'] = richness
        words[name]['hidden_size'] = hidden_size
        words[name]['weight_decay'] = weight_decay
        words[name]['dominance'] = dominance
        words[name]['category'] = category
        words[name]['net_seed'] = net_seed
        words[name]['data_seed'] = data_seed
        words[name]['ex_type'] = ex_type

        if line_type == 'output':
            # all but last char bc its \n
            out = list(map(float, entries[out_start:len(entries) - 1]))
            words[name]['stress'] = stress(out)
            out_at_tick[name][tick] = np.asarray(out)
            words[name]['avg_out_tick_' +
                        str(tick)] = np.average(np.asarray(out))
            if method == 1:  # change
                out = list(map(round, out))
            words[name]['out'] = np.asarray(out)
        else:  # target
            if '-' not in entries[out_start:len(entries) - 1]:
                targ = list(map(float, entries[out_start:len(entries) - 1]))
                words[name]['targ'] = np.asarray(targ)

        if tick in out_at_tick[name].keys() and (tick - 1) in out_at_tick[name].keys():
            prev_out = out_at_tick[name][tick-1]
            curr_out = out_at_tick[name][tick]
            act_change = np.average(abs(prev_out - curr_out))

            words[name]['avg_out_diff_tick_' + str(tick)] = act_change
            if act_change < st_threshold and 'st_thresh0.5' not in words[name].keys():
                words[name]['st_thresh0.5'] = tick

        if 'out' in words[name].keys() and 'targ' in words[name].keys():
            out = words[name]['out']
            targ = words[name]['targ']

            diff = np.sum(abs(out-targ))
            words[name]['diff_thresh0.5'] = diff
            if diff == 0 and 'rt_thresh0.5' not in words[name].keys():
                words[name]['rt_thresh0.5'] = tick
                words[name]['acc_thresh0.5'] = 1
                del words[name]['out']
                if 'targ' in words[name].keys():
                    del words[name]['targ']

        tick += 1
        count += 1

        prev_name = name

        if count == len(lines):
            words = prev_word_cleanup(words, name)

    df = save_to_csv(words, save_loc)
    print(df)
    # print(words['0_KOT']['rt_thresh0.5'], words['1_RON']['rt_thresh0.5'])


def draw_graphs(df, file_type, save_loc):
    name = 'Word' if file_type == 'train' else 'Non-Word'
    avg_out_diffs = []
    stderr_out_diffs = []
    avg_outs = []
    stderr_outs = []
    ticks = np.arange(20) + 1
    for col in df.columns:
        if 'avg_out_diff' in col:
            numpy_col = df[col].to_numpy()
            avg_of_col = np.average(numpy_col)
            stderr_of_col = stats.sem(numpy_col)
            avg_out_diffs.append(avg_of_col)
            stderr_out_diffs.append(stderr_of_col)
        elif 'avg_out' in col and 'avg_out_diff' not in col:
            numpy_col = df[col].to_numpy()
            avg_of_col = np.average(numpy_col)
            stderr_of_col = stats.sem(numpy_col)
            avg_outs.append(avg_of_col)
            stderr_outs.append(stderr_of_col)

    plt.figure(figsize=(8, 6), dpi=80)
    plt.errorbar(ticks, avg_out_diffs, yerr=stderr_out_diffs)
    plt.title(
        name + ' Averages of differences between output averages (tick_n - tick_(n-1))')
    plt.savefig(save_loc + '_diffavgs.png')

    plt.figure(figsize=(8, 6), dpi=80)
    plt.errorbar(ticks, avg_outs, yerr=stderr_outs)
    plt.title(name + ' Averages of output averages')
    plt.savefig(save_loc + '_outavgs.png')
    plt.show()


epoch_num = 620
base_path = '/Users/akathian/Desktop/School/akathian05@gmail.com/LinuxDesk/word-recognition/data/training_data/0seed_100hid_0TestRuns_800UpdateEpoch_0decay'

train_path = f'{base_path}/words_0seed_100hid_0decay_{epoch_num}epoch.txt'
test_path = f'{base_path}/nonwords_0seed_100hid_0decay_{epoch_num}epoch.txt'

train_save_loc = f'{base_path}/words_analysis.csv'
test_save_loc = f'{base_path}/nonwords_analysis.csv'

analyze(train_path, 1, 1, 1, 0.01, 'train', train_save_loc)
analyze(test_path, 1, 1, 1, 0.01, 'test', test_save_loc)

train_analysis = pd.read_csv(train_save_loc)
test_analysis = pd.read_csv(test_save_loc)

train_graph_save_loc = f'{base_path}/words_graph_analysis'
test_graph_save_loc = f'{base_path}/nonwords_graph_analysis'
draw_graphs(train_analysis, 'train', train_graph_save_loc)
draw_graphs(test_analysis, 'test', test_graph_save_loc)

# cols
# word degr netparams(From tr filename) acc rt
# min-binary-correct

# settling time
#   determine rt then figure out if correct
#   cant respond at first tick
#   take op at tick n, substract from n - 1 tick
#   take absolute value
#   take average of values
#   if average < threshold
#       this is rt
#       figure out if accurate (use rounding)
#       "criterion correct"
#   else end of trial
#       rt, acc = NaN

# compute average for every word, for every tick - plot this with error bar
# plot acc & rt in R
# put in regression

# only look at last tick
# take a look if mostly on vs mostly off
# how many -1 vs +1 before taking abs

# plot hsitogram of diff for all ex

# --------------------------------------------------------------
# high freq!!!!!!!!!! and stuff freq sem rich, cat dispersion
# rename diff to thresholded diff 0.5
# add accuracy column 1 or 0
# add rt settling time 0.01, acc settling time 0.01 (use same diff method)
# add non words - no accuracy, no rt for thresh diff 0.5
# include avg_tick_1 as NaNs


# lens resetFlagOnExample
# .initInput/Output


# generate graph like on the excel for each ex
# add non-words (diff colors)!
# add average of output at each tick
# add stress (polarity)
