import numpy as np
import re
import pickle
import feedparser

'''



'''







def loadDataSet():
    postList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0, 1, 0, 1, 0, 1]   # 1 for abusive, 0 fro not
    return postList, classVec


def createVovabularyList(dataset):
    vocabSet = set([])
    for document in dataset:
        vocabSet = vocabSet | set(document) # | is a union operation

    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word %s is not in the Vocabulary'%word)

    return returnVec

def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        # else:
        #     print('the word %s is not in the Vocabulary'%word)

    return returnVec




def trainNB0(trainMatrix, labels):
    number_of_samples = len(trainMatrix)
    number_of_words = len(trainMatrix[0])
    p_Abusive = sum(labels)/number_of_samples


    #防止因为没出现而造成0的现象，连乘之后导致结果为0
    p0_numerator = np.ones(number_of_words)
    p1_numerator = np.ones(number_of_words)
    '''
    为整数，假设有5个training document， 其中一个词每次都出现，则p0_numerator=6为了让分母与分子不同且分母永远比分子大，就让分母从
    2开始加
    '''
    p0_denominator = 2.0
    p1_denominator = 2.0

    for i in range(number_of_samples):
        if labels[i] == 1:
            #为所有1情况下，各个单词occurance各自之和

            p1_numerator += trainMatrix[i]
            #为所有w的occurance之和
            p1_denominator += sum(trainMatrix[i])

        else:
            p0_numerator += trainMatrix[i]

            p0_denominator += sum(trainMatrix[i])
    # 因为log单调递增切此处概率大于零，可以让log(A*B) = logA+logB，变为和的形式p(w1|c)*p(w2|c) = log(p(w1|c))+log(p(w2|c))

    p0_Vec = np.log(p0_numerator/p0_denominator)
    p1_Vec = np.log(p1_numerator/p1_denominator)

    return p0_Vec, p1_Vec, p_Abusive








def classifyNB(testVec, p0, p1, pA):
    p_0 = sum(testVec * p0) + np.log(pA)
    p_1 = sum(testVec * p1) + np.log(pA)

    if p_0>p_1:
        return 0
    else:
        return 1

def testingNB():
    # vocabulary = list()
    # temp = list()
    # for i in range(len(labels)):
    #     if labels[i] == 1:
    #         temp.append(postList[i])
    postList, labels = loadDataSet()
    vocabulary = createVovabularyList(postList)
    trainMatrix = list()

    for i in range(len(postList)):
        trainMatrix.append(setOfWords2Vec(vocabulary, postList[i]))

    p0, p1, pA = trainNB0(trainMatrix, labels)

    testDoc_1 = ['garbage', 'stupid', 'dog']
    testDoc_2 = ['popo', 'tsy', 'cute']

    testVec_1 = setOfWords2Vec(vocabulary, testDoc_1)
    testVec_2 = setOfWords2Vec(vocabulary, testDoc_2)

    print(classifyNB(testVec_1, p0, p1, pA))
    print(classifyNB(testVec_2, p0, p1, pA))


def getTokens(textString):

    textString = textString.strip()
    words = re.findall(r'\w*', textString)
    tokenList = [word.lower() for word in words if len(word)>2]
    # tokenList = list()
    # for word in words:
    #     if len(word)>3:
    #         tokenList.append(word)
    return tokenList

# with open('ham/1.txt', 'r') as file:
#     line = str(file.read())
#     tokens = getTokens(line)
#     print(tokens)


def testingNB_email():
    testSet = []
    trainingLabels = []
    testLabels = []
    trainingMatrix = []
    fullText = []

    for i in range(1, 26):

        with open('ham/%s.txt'%i, 'r') as file:
            line = str(file.read())
        tokens = getTokens(line)
        trainingMatrix.append(tokens)
        fullText.extend(tokens)
        trainingLabels.append(0)

        with open('spam/%s.txt' % i, 'r') as file:
            line = str(file.read())
        tokens = getTokens(line)
        trainingMatrix.append(tokens)
        trainingLabels.append(1)
        fullText.extend(tokens)
    vocabulary = createVovabularyList(trainingMatrix)
    for _ in range(10):
        index = int(np.random.uniform(len(trainingMatrix)+1)-1)

        testSet.append(trainingMatrix[index])
        testLabels.append(trainingLabels[index])
        del(trainingMatrix[index])
        del(trainingLabels[index])

    #now we get the training set and the testing set


    trainingSet = []
    for item in trainingMatrix:
        trainingSet.append(setOfWords2Vec(vocabulary, item))


    p0, p1, pA = trainNB0(np.array(trainingSet), np.array(trainingLabels))
    error = 0
    for i in range(len(testSet)):
        if classifyNB(np.array(setOfWords2Vec(vocabulary, testSet[i])), p0, p1, pA) != testLabels[i]:
            error += 1
            print('error document is: ', testSet[i])
    error_rate = error/len(testSet)
    return error_rate

# error_rate = testingNB_email()
# print('error rate is :', error_rate)



def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    import feedparser

    docList = []
    classList = []
    fullText = []

    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = getTokens(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = getTokens(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabulary = createVovabularyList(docList)
    top30words = calcMostFreq(vocabulary, fullText)
    for word in top30words:
        if word[0] in vocabulary:
            vocabulary.remove(word[0])

    training_index_Set = list(range(2*minLen))   #因为从两个源各去了minLen长度的内容, 此处需要用list把range转化，否则无法删除操作
    test_index_Set = []
    testClasses = []

    for i in range(20):
        index = int(np.random.uniform(0, len(training_index_Set)))
        test_index_Set.append(training_index_Set[index])
        del(training_index_Set[index])


    trainingMatix = []
    trainingClasses = []

    for index in training_index_Set:
        trainingMatix.append(bagOfWords2Vec(vocabulary, docList[index]))
        trainingClasses.append(classList[index])

    p0, p1, pA = trainNB0(trainingMatix, trainingClasses)
    error = 0
    for index in test_index_Set:
        if classList[index] != classifyNB(bagOfWords2Vec(vocabulary, docList[index]), p0, p1, pA):
            error += 1

    error_rate = float(error)/len(test_index_Set)

    print(error_rate)

    return p0, p1, vocabulary


def getTopWord(ny, sf):
    pSF, pNY, vocabulary = localWords(ny, sf)#ny -->feed1, sf -->feed0
    topSF = []
    topNY = []
    for i in range(len(pSF)):
        if pSF[i]>-6:
            topSF.append((vocabulary[i], pSF[i]))
        if pNY[i]>-6:
            topNY.append((vocabulary[i], pNY[i]))

    sortedSF = sorted(topSF, key=lambda item: item[1], reverse=True)
    print('SF*'*10)
    for item in sortedSF[:10]:
        print(item[0])


    print('NY*'*10)
    sortedNY = sorted(topNY, key=lambda item: item[1], reverse=True)
    for item in sortedNY[:10]:
        print(item[0])

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
#if we get rid of some stop words, then we can make the error rate lower



getTopWord(ny, sf)
'''
0.45
SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*
butt
married
all
great
from
hard
workout
man
could
face
NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*
home
did
any
there
breasts
wear
boy
love
well
see


'''


#
# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# print(ny['entries'][0]['summary'])







