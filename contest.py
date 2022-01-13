import numpy as np
import mnist
import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)

#import matplotlib.pyplot as plt
#from pylab import cm

SIZE = 28 * 28  #画像サイズ
NUM = 10000     #画像枚数
C = 10          #クラス数
M = 99          #中間層のノード数
X = np.loadtxt("le4MNIST_X.txt")
#X = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
#Y = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")
npz = np.load('np_savezA.npz')
W1 = npz['zw1']
b1 = npz['zb1']
W2 = npz['zw2']
b2 = npz['zb2']

def main():
    try: 
        var = pre_pro()
        x = input_layer(var)
        t = mid_join(x)
        y1 = relu(t)
        a = out_join(y1)
        y2 = softmax(a)
        y = after_pro(y2)
        print("Output is ", end="")
        print(y)
    except:
            print("Error")
    
    
    

def pre_pro(): #前処理
    i = input("Please input natural number from 0 to 9999: ")
    var = int(i)
    if var < 0 or var > 9999:
        print("Out of bound ", end="")
    return var


def input_layer(var): #入力層
    x = np.ravel(X[var])
    return x

def mid_join(x): #中間層への入力を計算する層（全結合層)
    global W1, b1
    t = W1@x + b1
    return t

def sigmoid(t): #シグモイド関数
    y1 = 1 / (1 + np.exp(-t))
    return y1

def relu(t): #ReLU
    y1 = np.maximum(0, t)
    return y1

def out_join(y1): #出力層への入力を計算する層（全結合層）
    global W2, b2
    a = W2@y1 + b2
    return a

def softmax(a): #ソフトマックス関数
    alpha = np.array([max(a)]*C)
    y2 = np.exp(a - alpha)/np.sum(np.exp(a - alpha))
    return y2

def after_pro(y2): #後処理
    y = np.argmax(y2)
    return y



if __name__ == "__main__":
    main()