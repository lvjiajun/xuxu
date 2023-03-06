# -*- coding: utf-8 -*-
# @Time : 2023/2/24 22:45
# @Author : cloudjumper
# @Email : lvjiajun@outlook.com
# @File : chat_gpt.py
# @Project : xuxu
from revChatGPT.V1 import Chatbot


token_list = list()
token_index = 0


def chatgpt_streamed(prompt: str):
    global token_index, token_list

    token_index = token_index % len(token_list)
    chatbot = Chatbot(config={
        "access_token": token_list[token_index]
    })
    prev_text = ""
    for data in chatbot.ask(
            prompt,
    ):
        message = data["message"][len(prev_text):]
        print(message, end="", flush=True)
        prev_text = data["message"]
    token_index += 1
    return prev_text


def chatgpt_single(prompt: str):
    global token_index, token_list

    token_index = token_index % len(token_list)
    chatbot = Chatbot(config={
        "access_token": token_list[token_index]
    })
    response = ""
    for data in chatbot.ask(
            prompt
    ):
        response = data["message"]
    return response


def init_chatbot(token: list):
    global token_list
    token_list = list(token)
