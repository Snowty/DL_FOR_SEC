# WebShell
WebShell是以ASP、PHP、JSP、或者CGI等文件形式存在的一种命令执行环境。
常见的WebShell检测方法主要有以下几种：

+ 静态检测，通过匹配特征码、特征值、危险操作函数来查找。但是只能查找已知的WebShell
+ 动态检测，检测执行时刻表现出来的特征，如数据库操作、敏感文件读取等
+ 语法检测，根据php语言扫描编译的实现方式，来实现关键危险函数的捕捉
+ 统计学检测，通过信息上、最长单词、重合指数、压缩比等


# 一些webshell数据集
+ [tennc](https://github.com/tennc/webshell)
+ [xl7dev](https://github.com/xl7dev/WebShell)
+ [tdifg](https://github.com/tdifg/WebShell)
+ [ysrc](https://github.com/ysrc/webshell-sample)
+ [xiaoxiaoleo](https://github.com/xiaoxiaoleo/xiao-webshell)
