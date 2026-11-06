#-*- coding : utf-32 -*-

import matplotlib.pyplot as plt

import CONFIG
import PATH

import numpy as np

def init_log(num_line=4):
    with open(PATH.TRAIN_LOG_PATH,'w+') as f:
        for i in range(0, num_line):
            f.write("[]" + "\n")
        f.flush()

def write_log(train_loss_list: list, train_metrics_list: list, val_loss_list: list, val_metrics_list: list):
    with open(PATH.TRAIN_LOG_PATH,'r') as f:
        newLog = []
        f.seek(0,0)
        try:
            oldLog = eval(f.readline())
            newLog.append(oldLog + train_loss_list)
            oldLog = eval(f.readline())
            newLog.append(oldLog + train_metrics_list)
            oldLog = eval(f.readline())
            newLog.append(oldLog + val_loss_list)
            oldLog = eval(f.readline())
            newLog.append(oldLog + val_metrics_list)

            
        except Exception as e:
            print(e)
            newLog.append(train_loss_list)
            newLog.append(train_metrics_list)
            newLog.append(val_loss_list)
            newLog.append(val_metrics_list)

    with open(PATH.TRAIN_LOG_PATH,'w+') as f:
        for item in newLog:
            f.write(str(item))
            f.write("\n")
        f.flush()

    return

def plt_drawing():
    with open(PATH.TRAIN_LOG_PATH,'r') as f:
        plt.plot(eval(f.readline()), label = 'loss')
        plt.plot(eval(f.readline()), label = 'r_square')
        plt.plot(eval(f.readline()), label = 'val_loss', linestyle="--")
        plt.plot(eval(f.readline()), label = 'val_r_square', linestyle="--")

    plt.xlabel('Epoch')
    plt.ylabel('Loss & Metrics')

    plt.ylim([0, 1])
    plt.legend(loc='right')
    plt.savefig(PATH.TRAIN_CURVEIMG_PATH, dpi=600)
    #plt.show()
    plt.clf()
    
    return