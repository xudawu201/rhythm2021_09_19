'''
Author: xudawu
Date: 2021-10-29 17:44:58
LastEditors: xudawu
LastEditTime: 2021-10-29 18:18:34
'''
import os #文件夹操作
import torch
class saveModel_model():
    def saveModel_model(model,optimizer,epoch,save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        state = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
        torch.save(state,save_dir)

    def loadModel_model(save_dir):
        checkpoint_model = torch.load(save_dir)
        # model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # start_epoch = checkpoint['epoch'] + 1
        return checkpoint_model
