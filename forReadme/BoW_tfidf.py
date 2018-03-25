#!/usr/bin/env python
# encoding: utf-8
#@author: stardustsky
#@file: validate.py
#@time: 2017/9/8 10:36
#@desc:

import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#词袋模型，这里的min_df取值为3，即该向量在整个payload中至少出现了三次
vec=CountVectorizer(min_df=3,ngram_range=(1,1))
content=[
    '<s[NULL]cript>alert(1)</s[NULL]cript>X</a>',
    '\'><video><source o?UTF-8?Q?n?error="alert(1)">',
    '\'><FRAMESET><FRAME RC=""+"javascript:alert(\'X\');"></FRAMESET>',
    '"></script>\'//<svg "%0Aonload=alert(1) //>',
    '"></script><img \'//"%0Aonerror=alert(1)// src>',
    'id%3Den%22%3E%3Cscript%3Ealert%28%22AKINCILAR%22%29%3C/script%3E',
    '?a%5B%5D%3D%22%3E%3Cscript%3Ealert%28document.cookie%29%3C/script%3E',
    '><iframe src="data:data:javascript:,% 3 c script % 3 e confirm(1) % 3 c/script %3 e">',
    '?mess%3D%3Cscript%3Ealert%28document.cookie%29%3C/script%3E%26back%3Dsettings1',
    'title%3D%3Cscript%3Ealert%28%22The%20Best%20XSSer%22%29%3C/script%3E',
    '<script charset>alert(1);</script charset>',
    '"><meta charset="x-mac-farsi">??script ??alert(1)//??/script ??',
    '</script><script>/*"/*\'/**/;alert(1)//</script>#',
]


# fit_transform():计算各个词语出现的次数
# get_feature_names():获取词袋中所有文本的关键字
# toarray():词频矩阵的结果
trans=TfidfTransformer()
tfidf=trans.fit_transform(vec.fit_transform(content))
print(vec.get_feature_names())
print(tfidf.toarray())
