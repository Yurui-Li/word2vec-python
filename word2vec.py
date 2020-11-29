#!/usr/bin/env python
# coding: utf-8

from math import exp,pow,sqrt
import numpy as np
import time
import argparse 


# ## (一)训练参数
# 注意，这些参数都是全局参数，包括以下参数
# 
# size: 对应代码中layer1_size， 表示词向量的维度，默认值是100。
# 
# train: 对应代码中train_file， 表示语料库文件路径。
# 
# save-vocab: 对应代码中save_vocab_file, 词汇表保存路径。
# 
# read-vocab: 对应代码中read_vocab_file， 表示已有的词汇表文件路径，直接读取，不用从语料库学习得来。
# 
# debug: 对应代码中debug_mode， 表示是否选择debug模型，值大于1表示开启，默认是2。开启debug会打印一些信息。
# 
# binary: 对应代码中全局变量binary，表示文件保存方式，1表示按二进制保存，0表示按文本保存，默认是0.
# 
# cbow: 对应代码中cbow， 1表示按cbow模型训练， 0表示按skip模式训练，默认是1。
# 
# alpha: 对应代码中alpha，表示学习率。skip模式下默认为0.025， cbow模式下默认是0.05。
# 
# output: 对应代码中output_file， 表示词向量保存路径。
# 
# window: 对应代码中window，表示训练窗口大小。默认是5
# 
# sample: 对应代码中sample，表示下采样阀值。
# 
# hs: 对应代码中hs， 表示按huffman softmax模式训练。默认是0， 表示不使用hs。
# 
# negative: 对应代码中negative， 表示按负采样模式训练， 默认是5。值为0表示不采用负采样训练；如果使用，值一般为3到10。
# 
# threads: 对应代码中num_threads，训练线程数，一般为12。
# 
# iter: 对应代码中iter，训练迭代次数，默认是5.
# 
# min-count: 对应代码中min_count，表示最小出现频率，低于这个频率的词会被移除词汇表。默认值是5
# 
# classes: 对应代码中classes，表示聚类中心数， 默认是0， 表示不启用聚类。
# 
# min-count: read



#定义一些全局变量
MAX_STRING = 100
EXP_TABLE_SIZE = 1000
MAX_EXP = 6
MAX_SENTENCE_LENGTH = 1000
MAX_CODE_LENGTH = 40
CLOCKS_PER_SEC = 1000


#初始化
binary = 0; cbow = 1; debug_mode = 2; window = 5; min_count = 5; num_threads = 12
layer1_size = 100; train_words = 0; word_count_actual = 0; iter_num = 5;file_size = 0; classes = 0
alpha = 0.025; sample = 1e-3;hs = 0;negative = 5; table_size = int(1e8)
vocab=[]
vocab_hash_size = 30000000  # Maximum 30 * 0.7 = 21M words in the vocabulary
vocab_hash=[-1]*vocab_hash_size
table=[0] * table_size
vocab_size=0
vocab_max_size = 1000
min_reduce= 1

syn0=[]
syn1=[]
syn1neg=[]

# ## (二）预生成expTable
# 
#   word2vec计算过程中用上下文预测中心词或者用中心词预测上下文，都需要进行预测；而word2vec中采用的预测方式是逻辑回归分类，需要用到sigmoid函数，具体函数形式为:
#   
#   $$\sigma (x) = \frac {1} {1+e^{-x}} = \frac {e^x} {1+e^x} $$
#   
#   在训练过程中需要用到大量的sigmoid值计算，如果每次都临时去算 $e^x$的值，将会影响性能；当对精度的要求不是很严格的时候，我们可以采用近似的运算。在word2vec中，将区间 \[-MAX_EXP, MAX_EXP\](代码中MAX_EXP默认值为6)等距划分为 EXP_TABLE_SIZE等份，并将每个区间的sigmoid值计算好存入到expTable中。在需要使用时，只需要确定所属的区间，属于哪一份，然后直接去数组中查找。 expTable初始化代码如下:

# In[6]:


#exp_Table
expTable=[0]*EXP_TABLE_SIZE
for i in range(len(expTable)):
    expTable[i]=exp((i/EXP_TABLE_SIZE * 2 - 1) * MAX_EXP) #Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1) #Precompute f(x) = x / (x + 1)


# ## （三）构建词汇库
# 
# 构建词汇库过程中，先判断是否已经有处理好现成的词汇库，有的话直接读取，没有的话再进行训练。
# 
# 词汇表训练过程分为以下几个步骤：
#   
#   1.读取一个单词，
#   
#   2.计算单词对应hash值，
#   
#   3.通过hash值得到单词在词汇表中索引，
#   
#   4.将单词加入到词汇表，
#   
#   5.对词汇表根据词频进行降序排序, 
#   
#   6.保存训练好的词汇表。
#   
#   依次介绍以上几个步骤。首先给出词汇表中每个词对应的结构体：

# In[7]:


class vocab_word(object):
    def __init__(self,cn=0,point="",word="",code="",codelen=0):
        self.cn=cn #词频
        self.point=point #记录huffman树中父节点索引， 自顶向下
        self.word=word #表示该单词
        self.code=code #表示Huffman编码表,记录父节点是左节点还是右节点
        self.codelenn=codelen #表示码值表长度


def ReadWord(fin):
    '''
    Reads a single word from a file, assuming space + tab + EOL to be word boundaries
    从文件中读取单个单词，假设单词之间通过空格或者tab键或者EOL键进行分割的
    Args:
        fin -- input stream
    Returns:
        word
    '''
    global vocab
    a=0
    word=""
    ch=fin.read(1) #先读一个字符,如果文件末尾，读到是空字符
    while ch!="":                                             
        if ord(ch) == 13: #如果是回车符 \r
            ch=fin.read(1)
            continue                                                                       
        if (ch == ' ') or (ch == '\t') or (ch == '\n'):     #代表一个单词结束的边界
            if a > 0:                                               
                if ch == '\n': #如果读到了单词但是遇到了换行符，
                    fin.seek(fin.tell()-1)    #退回到流中                   
                break
            else:
                ch=fin.read(1)
                continue  #开始读取时遇到空格或者tab键或者EOL键则继续
            if ch == '\n':                                       #仅仅读到了换行符
                word="</s>"                           #将</s>赋予给word
                return word
            else:
                continue

        word += ch
        a+=1
        if a > MAX_STRING: # Truncate too long words 截断
            word=word[:MAX_STRING]   
            break
        ch=fin.read(1)
    return word

def GetWordHash(word):
    '''
    Returns hash value of a word
    返回一个词对应的hash值
    Args:
        word -- str
    Return:
        hash_value -- int
    '''
    hash_value = 0
    for a in range(len(word)):
        hash_value = hash_value * 257 + ord(word[a])
    hash_value = hash_value % vocab_hash_size
    return hash_value

def SearchVocab(word):
    '''
    Returns position of a word in the vocabulary; if the word is not found, returns -1
    开放地址发得到词的位置，如果未找到返回-1
    Args:
        word -- str
        vocab_hash -- global
    Return:
        
    '''
    hash_value = GetWordHash(word) #获得索引
    while True:
        if vocab_hash[hash_value] == -1:
            return -1
        if word==vocab[vocab_hash[hash_value]].word:
            return vocab_hash[hash_value]
        hash_value = (hash_value + 1) % vocab_hash_size   #开放定址法
    return -1

def ReadWordIndex(fin):
    '''
    Reads a word and returns its index in the vocabulary
    Args:
        fin -- input stream
    Return
        index in the vocabulary
    '''                     
    word=ReadWord(fin) #从文件流中读取一个单词
    if word=="":  #读到文件末尾
        return -1
    return SearchVocab(word) #返回对应的词汇表中索引

def AddWordToVocab(word):
    '''
    Adds a word to the vocabulary
    将word加入到词汇表
    Args:
        word -- str
        vocab -- global
    Return:
        index
    '''
    global vocab
    global vocab_size
    global vocab_hash
    
    sample=vocab_word()
    if len(word) > MAX_STRING:
        word = word[:MAX_STRING]                               #规定每个word不超过MAX_STRING个字符
    sample.word = word                                        #结构体的word词
    vocab.append(sample)
    vocab_size+=1
    

    hash_value = GetWordHash(word)
    while vocab_hash[hash_value] != -1:
        hash_value = (hash_value + 1) % vocab_hash_size           #得到word实际对应的hash值
    vocab_hash[hash_value] = vocab_size - 1                    #通过hash值获得word在vocab中索引
    return vocab_size - 1                                   #返回单词对应索引

def VocabCompare(a,b):
    '''
    Used later for sorting by word counts
    构造一个比较器，用来排序，降序
    ps：实际并没有用到
    Args:
        a,b -- vocab_word
    Return:
        b.cn-a.cn
    '''
    return b.cn - a.cn

def SortVocab():
    '''
    Sorts the vocabulary by frequency using word counts
    '''
    a=0
    global vocab_size
    global vocab
    global vocab_hash
    global train_words
    # Sort the vocabulary and keep </s> at the first position
    vocab=[vocab[0]]+sorted(vocab[1:],key=lambda x:x.cn,reverse=True)
    vocab_hash=[-1 for _ in range(vocab_hash_size)]

    while a<vocab_size:
        # Words occuring less than min_count times will be discarded from the vocab
        #频率低于一定程度的词会被抛弃掉
        if vocab[a].cn < min_count and a!= 0:
            vocab_size-=1
            vocab.pop(a)
        else:
            # Hash will be re-computed, as after the sorting it is not actual
            #因为排序之后顺序打乱，会重新计算一次hash值
            hash_value=GetWordHash(vocab[a].word)
            while vocab_hash[hash_value] != -1:
                hash_value = (hash_value + 1) % vocab_hash_size
            vocab_hash[hash_value] = a
            train_words += vocab[a].cn
            a+=1

def SaveVocab():
    '''
    保存学习到的词汇文件表
    '''
    with open(save_vocab_file,"w",encoding="UTF-8") as fout:
        for i in range(vocab_size):
            fout.write(vocab[i].word+" "+str(vocab[i].cn)+"\n") #保存单词和词频


def ReduceVocab():
    '''
    Reduces the vocabulary by removing infrequent tokens
    对于频率小于min_reduce的词将会被裁剪掉
    '''
    global vocab_size
    global vocab
    global vocab_hash
    global min_reduce
    a = 0
    while a<vocab_size:
        if vocab[a].cn <= min_reduce:
            vocab.pop(a)
            vocab_size-=1
        else:
            a+=1

    #重新设置hash值
    vocab_hash=[-1 for _ in range(vocab_hash_size)]
    
    for a in range(vocab_size):
      #Hash will be re-computed, as it is not actual
        hash_value = GetWordHash(vocab[a].word)
        while vocab_hash[hash_value] != -1:
            hash_value = (hash_value + 1) % vocab_hash_size
        vocab_hash[hash_value] = a

    min_reduce+=1     #每次裁剪之后都会提高最低频率数


def LearnVocabFromTrainFile():
    '''
    整合上面的文件操作
    '''
    global train_words
    global vocab
    global file_size

    index_of_ring=AddWordToVocab("</s>")  #将'</s>'添加到词汇表，换行符就是用这个表示
    with open(train_file, "r",encoding="UTF-8") as fin:
        word=ReadWord(fin)
        while word!="":
            train_words+=1
            i = SearchVocab(word)   #查找该词的位置
            if i == -1:    #还未加入到词汇表                   
                a = AddWordToVocab(word)
                vocab[a].cn = 1
            else:
                vocab[i].cn+=1  #已经加入到词汇表
            if vocab_size > vocab_hash_size * 0.7:
                ReduceVocab()     #裁剪词操作
            word=ReadWord(fin)
        file_size = fin.tell() 
    SortVocab() #排序
    

# ## （四）初始化网络
# 初始化网络包括以下几个过程：
#   
#   1.初始化网络参数 
#   
#   2.构建哈夫曼树
#   
#   3.初始化负采样概率表

# ### 1.初始化网络参数
# 
# 网络中的参数主要包括syn0,syn1和syn1neg。
# 
# syn0: 我们需要得到的词向量，源码中使用一个real(float)类型的一维数组表示，注意是一个一维数组！
#       容量大小为vocab_size * layer1_size，即 词汇量 * 词向量维度。
# 
# syn1: huffman树中，包括叶子节点和非叶子节点。叶子节点是对应的是词汇表中的单词，而非叶子节点是在构造huffman树过程中
#       生成的路径节点。syn1表示的就是huffman树中的非叶子节点向量，其维度和词向量维度是一样的，共有(n-1)个非叶子节点，
#       n表示词汇表中单词量。注意，syn1也是一个一维real(float)数组，容量为 vocab_size * layer1_size
#       
# syn1neg: 这是单词的另一个向量表示，之前看斯坦福自然语言处理视频中有提到过每个单词会训练出两个向量，现在看来的确是这    
#          样，不过是通过negative方式训练才有。这个向量是用于负采样模式优化时需要的变量。也是一个一维的float数组，
#          大小是 vocab_size * layer1_size。
#       
# syn0的每个值的范围为:\[$\frac {−0.5} {m}$,$\frac {0.5} {m}$\]，m表示向量维度；syn1初始化为0；syn1neg也初始化为0.


def InitNet():
    '''
    初始化网络
    '''
#这部分代码是仿照源码改写的，但在python下效率过低,考虑到源码初始化syn0的方式服从均匀分布，通过numpy初始化
#     long next_random = 1
#     #创建syn0为一维数组，长度为vocab_size * layer1_size ,也就是每个词汇对应一个layer1_size的向量
#     syn0=[0] * vocab_size * layer1_size
#     #对syn0中每个词对应的词向量进行初始化
#     for a in range(vocab_size):
#         for b in range(layer1_size):
#             next_random = next_random * 25214903917 + 11 #生成一个很大的数
#             syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / 65536 )- 0.5) / layer1_size #& 0xFFFF表示截断为[0, 65536]
    global syn0
    global syn1
    global syn1neg
    syn0=np.random.uniform(low=-0.5/layer1_size, high=0.5/layer1_size, size=vocab_size * layer1_size) #一般能在5秒内完成

    #如果采用huffman softmax构造，那么需要初始化syn1，长度为vocab_size * layer1_size，每个词对应一个
    if hs:
        syn1 = np.zeros(vocab_size * layer1_size)

    #如果采用负采样进行训练，那么久初始化syn1neg，长度为vocab_size * layer1_size ，每个词对应一个
    if negative>0:
        syn1neg = np.zeros(vocab_size * layer1_size) 

    #建huffman softmax需要的哈夫曼树
    CreateBinaryTree()


# ## 2.构建哈夫曼树
def CreateBinaryTree():
    '''
    Create binary Huffman tree using the word counts;
    Frequent words will have short uniqe binary codes
    构造哈夫曼树
    '''
    global vocab
    point=[0]*MAX_CODE_LENGTH
    code=[0]*MAX_CODE_LENGTH
    #构建数组，长度vocab_size * 2 + 1)
    #因为hufuman树的特性，所以总结点数是2 * n + 1, 其中n是节点数, 此处应该有错误，是2 * n - 1才对
    count = [v.cn for v in vocab]       #count存储节点对应频率,前半部分初始化为词频
    count += [1e15]*(vocab_size + 1)    #后半部分设为无穷
    binary = [0] * (vocab_size * 2 + 1)      #binary记录每个节点是左节点还是右节点
    parent_node = [0] * (vocab_size * 2 + 1) #parent_node记录父节点位置

    pos1 = vocab_size - 1
    pos2 = vocab_size
    # Following algorithm constructs the Huffman tree by adding one node at a time
    #如同天才般的代码，一次遍历就构造好了huffuman树, 注意,这个a还代表了一种顺序，所有count值由小到大的顺序
    for a in range(vocab_size - 1):
        # First, find two smallest nodes 'min1, min2',注意vocab中的词是已经按照cn排好序的了,是按照降序排列的
        # pos1表示取最原始的词对应的词频,而pos2表示取合并最小值形成的词频
        # 连续两次取，两次取的时候代码操作时一模一样的
        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min1i = pos1
                pos1-=1
            else:
                min1i = pos2
                pos2+=1
        else:
            min1i = pos2
            pos2+=1
        
        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min2i = pos1
                pos1-=1
            else:
                min2i = pos2
                pos2+=1
        else:
            min2i = pos2
            pos2+=1

        count[vocab_size + a] = count[min1i] + count[min2i]
        parent_node[min1i] = vocab_size + a                   #记录好合并形成的父节点的位置
        parent_node[min2i] = vocab_size + a
        binary[min2i] = 1                                     #左为0,右为1

    # Now assign binary code to each vocabulary word
    # 建好了hufuman树之后，就需要分配code了，注意这个hufuman树是用一个数组来存储的，并不是我们常用的指针式链表
    for a in range(vocab_size):
        b = a
        i = 0
        points=[];codes=[]
        while True:
            code[i] = binary[b]                                 #对于每个节点，自底向上得到code值，通过每个节点的binary来实现
            point[i] = b                                        #point记录节点到根节点经过的节点的路径
            i+=1
            b = parent_node[b]
            if b == vocab_size * 2 - 2:
                break

        vocab[a].codelen = i                                  #记录词对应的码值的长度
        points.append(vocab_size - 2)                   #最大值作为根节点
        for b in range(i-1,0,-1):
            codes.append(code[b])                  #倒序过来，自顶向下
            points.append(point[b] - vocab_size) #注意这个索引对应的是huffman树中的非叶子节点，对应syn1中的索引， 因为非叶子节点都是在vocab_size * 2 + 1 的后(vocab_size + 1)个
        vocab[a].point=points
        vocab[a].code=codes

# ## 3.初始化负采样概率表
#   如果是采用负采样的方法，此时还需要初始化每个词被选中的概率。在所有的词构成的词典中，每一个词出现的频率有高有低，我们希望，对于那些高频的词，被选中成为负样本的概率要大点，同时，对于那些出现频率比较低的词，我们希望其被选中成为负样本的频率低点。

def InitUnigramTable():
    '''
    生成负采样的概率表
    '''
    global table
    power = 0.75
    train_words_pow = 0.

    #pow(x, y)计算x的y次方;train_words_pow表示总的词的概率，不是直接用每个词的频率，而是频率的0.75次方幂
    for a in range(vocab_size):
        train_words_pow += pow(vocab[a].cn, power);  
    i = 0
    d1 = pow(vocab[i].cn, power) / train_words_pow;
    #每个词在table中占的小格子数是不一样的，频率高的词，占的格子数显然多
    for a in range(int(table_size)):
        table[a] = i
        if (a / table_size) > d1:
            i+=1
            d1 += pow(vocab[i].cn, power) / train_words_pow
        if i >= vocab_size:
            i = vocab_size - 1

# ## (五）模型训练
#   关于word2vec的CBOW和SKIP模型原理，强力推荐大神的博客讲解，虽然有错误细节，但是大体思想都是正确的。
# 首先定义了几个重要的变量，变量解释如下:
# 
# last_word： 当前窗口正在训练的词的索引。
# 
# sentence_length: 当前训练的句子的长度
# 
# sentence_position: 当前中心词在句子中的位置
# 
# sen: 数组，存的是句子中每个词在词汇表中的索引
# 
# neu1: 是cbow模式下映射层对应的上下文向量表示，为上下文中所有词向量的平均值
# 
# neu1e: 因为skip模式下，映射层向量就是输入层向量的复制，所以neu1e仅仅用来记录上下文词对输入层的梯度。

def TrainModelThread(vid):
    global word_count_actual
    global syn0
    global syn1
    global syn1neg
    global alpha
    
    last_word_count = 0
    sentence_length = 0
    next_random = vid
    word_count = 0
    sentence_position = 0
    sen=[0] * (MAX_SENTENCE_LENGTH + 1)

    local_iter = iter_num
    fi = open(train_file, "r");
    fi.seek(int(file_size / num_threads * vid))
    while True:
        if word_count - last_word_count > 10000:
            word_count_actual += word_count - last_word_count
            last_word_count = word_count
            if debug_mode > 1:
                now=time.time()
                print("\rAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  " %( alpha, \
                word_count_actual / (iter_num * train_words + 1) * 100, \
                word_count_actual / ((now - start + 1) / CLOCKS_PER_SEC * 1000)))
            #学习率衰减
            alpha = starting_alpha * (1 - word_count_actual / (iter_num * train_words + 1))
            if alpha < starting_alpha * 0.0001:
                alpha = starting_alpha * 0.0001
        #每次读取一条句子，记录好句子中每个词在词汇表中对应的索引。
        #如果启用了下采样，则会随机的跳过一些词，会随机的丢弃频繁的单词，同时保持顺序不变
        if sentence_length == 0:
            while True:
                word = ReadWordIndex(fi)
                if word == -1:
                    if fi.read(1)=="":
                        break
                    continue
                word_count+=1
                if word == 0:break #遇到换行符，则直接跳出来，第一个词'</s>'代表换行符
                #The subsampling randomly discards frequent words while keeping the ranking same
                #下采样随机丢弃频繁的单词，同时保持排名相同，随机跳过一些词的训练
                if sample > 0:
                    ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
                    next_random = next_random * 25214903917 + 11;
                #频率越大的词，对应的ran就越小，越容易被抛弃，被跳过
                if ran < (next_random & 0xFFFF) / 65536:
                    continue

                sen[sentence_length] = word  #当前句子包含的词，存的是索引
                sentence_length+=1  #句子实际长度，减去跳过的词
                if sentence_length >= MAX_SENTENCE_LENGTH:
                    break
            sentence_position = 0
    
        #    
        if fi.read(1)=="" or (word_count > train_words / num_threads):
            word_count_actual += word_count - last_word_count;
            local_iter-=1
            if local_iter == 0:break   #训练结束
            word_count = 0
            last_word_count = 0 
            sentence_length = 0
            fi.seek(int(file_size / num_threads * vid))
            continue;

        word = sen[sentence_position]
        if word == -1:continue
        #neu1 = [0] * layer1_size
        neu1 = np.zeros(layer1_size)
        #neu1e = [0] * layer1_size
        neu1e = np.zeros(layer1_size)
        next_random = next_random * 25214903917 + 11;
        b = next_random % window
    
        if cbow:  #train the cbow architecture
          # in -> hidden
            cw = 0
            #随机取一个词word，然后计算该词上下文词对应的向量的各维度之和
            for a in (b,window * 2 + 1 - b):
                if a != window:
                    c = sentence_position - window + a
                    if c < 0:continue
                    if c >= sentence_length:continue
                    last_word = sen[c] #获得senten中第c个词的索引
                    if last_word == -1:continue
                    #注意syn0是一维数组，不是二维的，所以通过last_word * layer1_size来定位某个词对应的向量位置
                    #last_word表示上下文中上一个词
                    neu1 += syn0[last_word * layer1_size:(last_word + 1) * layer1_size]
                    cw+=1
               
            if cw:
                #上下文表示是所有词对应词向量的平均值
                neu1 /= cw
                if hs:
                    for d in range(vocab[word].codelen):
                        f = 0;
                        l2 = vocab[word].point[d] * layer1_size #索引到该词在数组偏移量
                        # Propagate hidden -> output
                        #syn1也是一维数组，不同词对应的位置需要偏移量l2确定
                        f += np.sum(neu1 * syn1[l2:(l2+layer1_size)])
                        if f <= -MAX_EXP:
                            continue  #f值不属于[-MAX_EXP, MAX_EXP]
                        elif f >= MAX_EXP:
                            continue
                        else:
                            #查看f属于第几份，((f + MAX_EXP) / (2 * MAX_EXP)) * EXP_TABLE_SIZE
                            f = expTable[int((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                        # 'g' is the gradient multiplied by the learning rate
                        g = (1 - vocab[word].code[d] - f) * alpha #需要推导,得到这个梯度比例
                        # Propagate errors output -> hidden
                        neu1e += g * syn1[l2:(l2+layer1_size)]  #这个部分才是最终梯度值
                        # Learn weights hidden -> output
                        syn1[l2:(l2+layer1_size)] += g * neu1  #加上梯度值，更新syn1

                # NEGATIVE SAMPLING
                if negative > 0:
                    for d in range(negative + 1):
                        if d == 0: #一个正样本
                            target = word
                            label = 1
                        else: 
                            #随机挑选一个负样本，负样本就是除中心词以外的所有词
                            next_random = next_random * 25214903917 + 11
                            target = table[int(next_random >> 16) % table_size]
                            #next_random = np.random.randint(table_size)
                            #target = table[next_random % table_size]
                            if target == 0:
                                #如果target为0，这个等式保证不为0
                                target = next_random % (vocab_size - 1) + 1
                            if target == word:
                                continue #正样本则跳过
                            label = 0
                        #负采样实际会为每个词生成两个向量
                        l2 = target * layer1_size
                        f = 0
                        f += np.sum(neu1 * syn1neg[l2:(l2+layer1_size)])
                        if f > MAX_EXP:
                            g = (label - 1) * alpha
                        elif f < -MAX_EXP:
                            g = (label - 0) * alpha
                        else:
                            g = (label - expTable[int((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha
                        neu1e += g * syn1neg[l2:(l2+layer1_size)]
                        syn1neg[l2:(l2+layer1_size)] += g * neu1

                # hidden -> in 更新输入层的词向量
                for a in range(b,window * 2 + 1 - b):
                    if a != window:
                        c = sentence_position - window + a
                        if c < 0:
                            continue
                        if c >= sentence_length:
                            continue
                        last_word = sen[c]
                        if last_word == -1:
                            continue
                        syn0[last_word * layer1_size:(last_word + 1)* layer1_size] += neu1e

        else: #train skip-gram
            #还是保证一个2 * window大小上下文，但是中心词左右并不一定刚好都是window个，根据b确定
            for a in range(b,window * 2 + 1 - b):
                if a != window:
                    c = sentence_position - window + a #c表示上下文的当前遍历位置
                if c < 0:
                    continue
                if c >= sentence_length:
                    continue
                last_word = sen[c]
                if last_word == -1:
                    continue
                l1 = last_word * layer1_size
                neu1e = 0 * neu1e
                # HIERARCHICAL SOFTMAX
                if hs:
                    for d in range(vocab[word].codelen):
                        f = 0;
                        l2 = vocab[word].point[d] * layer1_size;
                        # Propagate hidden -> output
                        f += np.sum(syn0[l1:(l1+layer1_size)] * syn1[l2:(l2+layer1_size)])
                        if f <= -MAX_EXP:
                            continue
                        elif f >= MAX_EXP:
                            continue
                        else:
                            f = expTable[int((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                        # 'g' is the gradient multiplied by the learning rate
                        g = (1 - vocab[word].code[d] - f) * alpha;
                        # Propagate errors output -> hidden
                        neu1e += g * syn1[l2:(l2+layer1_size)]
                        # Learn weights hidden -> output
                        syn1[l2:(l2+layer1_size)] += g * syn0[l1:(l1+layer1_size)]

                # NEGATIVE SAMPLING
                if negative > 0:
                    for d in range(negative + 1):
                        if d == 0:
                            target = word
                            label = 1
                        else:
                            next_random = next_random * 25214903917 + 11
                            target = table[int(next_random >> 16) % table_size]
                            #next_random = np.random.randint(table_size)
                            #target = table[next_random % table_size]
                            if target == 0:
                                target = next_random % (vocab_size - 1) + 1
                            if target == word:
                                continue
                            label = 0
                        l2 = target * layer1_size
                        f = 0
                        f += np.sum(syn0[l1:(l1+layer1_size)] * syn1neg[l2:(l2+layer1_size)])
                        if f > MAX_EXP:
                            g = (label - 1) * alpha
                        elif f < -MAX_EXP:
                            g = (label - 0) * alpha
                        else:
                            g = (label - expTable[int((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha
                        neu1e += g * syn1neg[l2:(l2+layer1_size)]
                        syn1neg[l2:(l2+layer1_size)] += g * syn0[l1:(l1+layer1_size)]

                # Learn weights input -> hidden
                syn0[l1:(l1+layer1_size)] += neu1e

        sentence_position += 1
        if sentence_position >= sentence_length:
            sentence_length = 0
            continue

    fi.close()

def tips():
    print("WORD VECTOR estimation toolkit v 0.1c\n\n");
    print("Options:\n");
    print("Parameters for training:\n");
    print("\t-train <file>\n");
    print("\t\tUse text data from <file> to train the model\n");
    print("\t-output <file>\n");
    print("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    print("\t-size <int>\n");
    print("\t\tSet size of word vectors; default is 100\n");
    print("\t-window <int>\n");
    print("\t\tSet max skip length between words; default is 5\n");
    print("\t-sample <float>\n");
    print("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    print("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    print("\t-hs <int>\n");
    print("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    print("\t-negative <int>\n");
    print("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    print("\t-threads <int>\n");
    print("\t\tUse <int> threads (default 12)\n");
    print("\t-iter <int>\n");
    print("\t\tRun more training iterations (default 5)\n");
    print("\t-min_count <int>\n");
    print("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    print("\t-alpha <float>\n");
    print("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    print("\t-classes <int>\n");
    print("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    print("\t-debug <int>\n");
    print("\t\tSet the debug mode (default = 2 = more info during training)\n");
    print("\t-binary <int>\n");
    print("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    print("\t-save_vocab <file>\n");
    print("\t\tThe vocabulary will be saved to <file>\n");
    print("\t-read_vocab <file>\n");
    print("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    print("\t-cbow <int>\n");
    print("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    print("\nExamples:\n");
    print("python word2vec.py -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");

if __name__=="__main__":
    #解析参数
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("-size",default=100,type=int,required=False,help="set size of word vectors; default is 100")
    parser.add_argument("-train", default="text8", type=str,required=False, help="Use text data from <file> to train the model")
    parser.add_argument("-cbow", default=1, type=int, required=False,                        help="Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)")
    parser.add_argument("-save_vocab", default="", type=str,help="The vocabulary will be saved to <file>")
    parser.add_argument("-read_vocab", default="", type=str, required=False,                         help="The vocabulary will be read from <file>, not constructed from the training data")
    parser.add_argument("-debug", default=2, type=int, required=False,                         help="set the debug mode (default = 2 = more info during training)")
    parser.add_argument("-binary", default=0, type=int, required=False,                         help="Save the resulting vectors in binary moded; default is 0 (off)")
    
    parser.add_argument("-alpha", default=0.025, type=float, required=False,                         help="Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW")
    parser.add_argument("-output", default="vectors.txt", type=str, required=False,                         help="Use <file> to save the resulting word vectors / word clusters")
    parser.add_argument("-window", default=5, type=int, required=False,                         help="Set max skip length between words; default is 5")
    parser.add_argument("-sample", default=1e-3, type=float, required=False,                         help="Set threshold for occurrence of words. Those that appear with higher frequency in the training data                         will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)")
    parser.add_argument("-hs", default=0, type=int, required=False, help="Use Hierarchical Softmax; default is 0 (not used)")
    parser.add_argument("-negative", default=5, type=int, required=False,                         help="Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)")
    parser.add_argument("-threads", default=12, type=int, required=False, help="Use <int> threads (default 12)")
    parser.add_argument("-iter", default=5, type=int, help="Run more training iterations (default 5)")
    parser.add_argument("-min_count", default=5, type=int, required=False,                         help="This will discard words that appear less than <int> times; default is 5")
    parser.add_argument("-classes", default=0, type=int, required=False,                         help="Output word classes rather than word vectors; default number of classes is 0 (vectors are written)")
    

    args = parser.parse_args()
    if args.cbow:
        args.alpha = 0.05

    print("args:\n" + args.__repr__())
    #将参数赋值
    layer1_size = args.size
    train_file = args.train
    save_vocab_file = args.save_vocab
    read_vocab_file = args.read_vocab
    debug_mode = args.debug
    binary = args.binary
    cbow = args.cbow
    alpha = args.alpha
    output_file = args.output
    window = args.window
    sample = args.sample
    hs = args.hs
    negative = args.negative
    num_threads = args.threads
    iter = args.iter
    min_count = args.min_count
    classes = args.classes
    
    #训练模型
    #因为python的多线程代码需要放在if __name__=="__main__"下，所以将源码中TrainModel() 功能拆解到此
    print("Starting training using file %s\n" % train_file)
    starting_alpha = alpha #设置学习率
    
    #获得词汇表，如果已经有直接读，否则学
    if read_vocab_file != "":
        ReadVocab()
    else:
        LearnVocabFromTrainFile()
    
    if save_vocab_file != "":
        SaveVocab()

    #初始化网络参数
    InitNet()
    #如果是使用负采样，那么需要负采样概率表
    if negative > 0:
        InitUnigramTable()
        
    #计时
    start = time.time() 
    
    #多线程训练
    TrainModelThread(1)
    
    #是否以二进制写入
    if binary:
        fo = open(output_file, "wb")
    else:
        fo = open(output_file, "w")
        
    #classes判断是否使用kmean聚类，为0表示否
    if classes == 0:
        # Save the word vectors
        print("%d %d\n" % (vocab_size, layer1_size),file=fo) #word2vec向量首行内容是 词汇量，向量维度
        for a in range(vocab_size):
            print("%s " % vocab[a].word,file=fo) #每行以单词开始，空格隔开
            vector = [str(i) for i in syn0[a * layer1_size:(a + 1) * layer1_size]]
            fo.write(" ".join(vector) + "\n")
            
    else:
        # Run K-means on the word vectors
        pass

    fo.close()


