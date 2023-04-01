'''
Author: xudawu
Date: 2023-03-04 10:57:44
LastEditors: xudawu
LastEditTime: 2023-03-31 20:03:07
'''

import openai

# 存储为txt文件，以追加方式
def saveInTxtFileByAppend_txt(fileName, content_str):
    file_txt = open(fileName, mode='a', encoding='utf-8')
    file_txt.write(content_str)  # write 写入
    file_txt.write('\r\n') # 写完一次换行
    file_txt.close()  # 关闭文件
    
openai.api_key = "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" 

class ChatGPT:
    def __init__(self,model_str,conversation_list=[]) -> None:
        # 初始化对话列表，可以加入一个key为system的字典，有助于形成更加个性化的回答
        # self.conversation_list = [{'system':'你是一个非常友善的助手'}]
        self.conversation_list = []
        self.model_str = model_str

    # 提示chatgpt
    def ask(self,prompt,count_int):
        # 对话10次之后清除最开始的对话
        if len(self.conversation_list) > 8:
            self.conversation_list.pop(0)
        # 获取回答并且添加到上下文列表中
        self.conversation_list.append({"role":"user","content":prompt})
        response = openai.ChatCompletion.create(model=self.model_str,messages=self.conversation_list)
        answer = response.choices[0].message['content']
        # 下面这一步是把chatGPT的回答也添加到对话列表中，这样下一次问问题的时候就能形成上下文了
        self.conversation_list.append({"role":"assistant","content":answer})
        # 返回问题和回答
        return prompt,answer

# 实例化模型
model_str='gpt-3.5-turbo'
Chat_text=ChatGPT(model_str)
# 查看有哪些模型
# print(openai.Model.list())

# askQuestion_str='人工智能方向有什么可以写的论文题目'
# Chat_text.ask(askQuestion_str)

# 问答计数
count_int=1
while True:
    print(' #'*10)
    print('如果要退出对话请输入‘退出对话’')
    askQuestion_str = input("请继续输入问题：")
    if askQuestion_str == "退出对话":
        break
    # 获取ChatGPT回答
    askQuestion_str,answer_str=Chat_text.ask(askQuestion_str,count_int)
    # 打印问答
    thisAsk_str='问题'+str(count_int)+askQuestion_str
    print(thisAsk_str)
    print(answer_str)
    # 存储问答
    fileName_str='ChatGptAsk.txt'
    saveInTxtFileByAppend_txt(fileName_str,thisAsk_str)
    saveInTxtFileByAppend_txt(fileName_str,answer_str)
    # 问题数加一
    count_int=count_int+1