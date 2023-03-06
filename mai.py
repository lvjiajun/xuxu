# -*- coding: utf-8 -*-
# @Time : 2023/3/4 23:24
# @Author : cloudjumper
# @Email : lvjiajun@outlook.com
# @File : mai.py
# @Project : xuxu
if __name__ == '__main__':
    import socket

    # 函数 gethostname() 返回当前正在执行 Python 的系统主机名
    res = socket.gethostbyname(socket.gethostname())
    print(res)
