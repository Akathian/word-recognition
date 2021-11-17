import numpy as np
import pandas as pd
import math


def analyze(file, method, measure, in_format, st_threshold):
    f = open(file, 'r')
    lines = f.readlines()
    count = 0
    words = dict()
    prev_name = ''
    out_at_tick = dict()
    for line in lines:
        if in_format == 1:  # akathian project format
            entries = line.split(' ')
            out_start = 4
            name, epoch, tick, line_type = entries[:out_start]
            tick = int(tick)
        elif in_format == 2:  # Ian project formet
            entries = line.split('|')
            output = entries[-1].split(' ')
            out_start = 1
            epoch, name = entries[:2]
            line_type = output[0]

        if count == 0:
            prev_name = name

        if name not in out_at_tick.keys():
            out_at_tick[name] = dict()

        # next word is presented
        if prev_name != name:
            if in_format == 2:
                tick = 0
            del words[prev_name]['out']
            del words[prev_name]['targ']
            if 'rt' not in words[prev_name].keys():
                words[prev_name]['rt'] = float("NaN")

            if 'st' not in words[prev_name].keys():
                words[name]['st'] = float("NaN")

        if name not in words.keys():
            words[name] = dict()
        if line_type == 'output':
            # all but last char bc its \n
            out = list(map(float, entries[out_start:len(entries) - 1]))
            out_at_tick[name][tick] = np.asarray(out)
            if method == 1:  # change
                out = list(map(round, out))
            words[name]['out'] = np.asarray(out)
        else:  # target
            targ = list(map(float, entries[out_start:len(entries) - 1]))
            words[name]['targ'] = np.asarray(targ)

        if tick in out_at_tick[name].keys() and (tick - 1) in out_at_tick[name].keys():
            prev_out = out_at_tick[name][tick-1]
            curr_out = out_at_tick[name][tick]
            act_change = np.average(abs(prev_out - curr_out))

            words[name]['avg_at_tick_' + str(tick)] = act_change
            if act_change < st_threshold and 'st' not in words[name].keys():
                words[name]['st'] = tick

        if 'out' in words[name].keys() and 'targ' in words[name].keys():
            out = words[name]['out']
            targ = words[name]['targ']
            diff = np.sum(abs(out-targ))
            words[name]['diff'] = diff
            if diff == 0 and 'rt' not in words[name].keys():
                words[name]['rt'] = tick
                del words[name]['out']
                del words[name]['targ']

        tick += 1
        count += 1

        prev_name = name

        if count == len(lines):
            del words[name]['out']
            del words[name]['targ']
            if 'rt' not in words[name].keys():
                words[name]['rt'] = float("NaN")

            if 'st' not in words[name].keys():
                words[name]['st'] = float("NaN")

    for key in words.keys():
        row = [key]
        row.append(words[key]['diff'])
        row.append(words[key]['rt'])
        row.append(words[key]['st'])
        for i in range(1, 20):
            print(key)
            row.append(words[key]['avg_at_tick_' + str(i+1)])
        words[key] = row

    print(words['1_RON'])
    ticks = []
    for i in range(1, 20):
        ticks.append('avg_at_tick_' + str(i+1))

    cols = ['word', 'diff', 'rt', 'st'] + ticks
    print('len cols', len(cols))
    df = pd.DataFrame.from_dict(
        words,  columns=cols, orient='index')
    df.to_csv('analysis.csv', index=False)
    print(df)
    # print(words['0_KOT']['rt'], words['1_RON']['rt'])


analyze('training_data/tr_0seed_100hid_0decay_605epoch.txt', 1, 1, 1, 0.01)

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
