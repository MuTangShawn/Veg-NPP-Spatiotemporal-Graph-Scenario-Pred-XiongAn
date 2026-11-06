# coding=gbk

import os

import DataProcess
import MakeDataset
import TrainModel
import DrawLog
import ModelPred

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def ConfigInput():
    flag_Train, flag_Draw, flag_Pred = 'A', 'A', 'A'

    while(not(flag_Train == 'Y' or flag_Train == 'N')):
        #flag_Train = input(r"是否对模型进行训练？(Y/N):")
        flag_Train = input(r"Train the model? (Y/N):")
        if(len(flag_Train) == 0): continue
        if(ord(str(flag_Train)[0]) > 90):
            flag_Train = chr(ord(flag_Train[0]) - 32) # 小写转大写
    while(not(flag_Draw == 'Y' or flag_Draw == 'N')):
        #flag_Draw = input(r"是否输出训练情况图表？(Y/N):")
        flag_Draw = input(r"Output training situation chart? (Y/N):")
        if(len(flag_Draw) == 0): continue
        if(ord(str(flag_Draw)[0]) > 90):
            flag_Draw = chr(ord(flag_Draw[0]) - 32)
    while(not(flag_Pred == 'Y' or flag_Pred == 'N')):
        #flag_Pred = input(r"是否使用模型进行输出？(Y/N):")
        flag_Pred = input(r"Use models for inference? (Y/N):")
        if(len(flag_Pred) == 0): continue
        if(ord(str(flag_Pred)[0]) > 90):
            flag_Pred = chr(ord(flag_Pred[0]) - 32)
    
    return flag_Train, flag_Draw, flag_Pred

def main():
    flag_Train, flag_Draw, flag_Pred = ConfigInput()

    if(flag_Train == 'Y'):
        train_data_list, val_data_list, test_data_list = DataProcess.org_data_process()
        train_loader, val_loader, test_loader = MakeDataset.make_dataset(train_data_list, val_data_list, test_data_list)
        history = TrainModel.train(train_loader, val_loader, test_loader)
    if(flag_Draw == 'Y'):
        DrawLog.plt_drawing()
    if(flag_Pred == 'Y'):
        ModelPred.model_predict()

    return

if(__name__ == "__main__"):
    main()
