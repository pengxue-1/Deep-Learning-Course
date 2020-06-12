import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()

boston_housing=tf.keras.datasets.boston_housing
(train_x,train_y),(test_x,test_y)=boston_housing.load_data()

x_train=train_x[:,5]
y_train=train_y
x_test=test_x[:,5]
y_test=test_y

#设置超参数
learn_rate=float(input("请输入学习率："))
iter=2000
display_step=int(input("请输入迭代数："))

#设置模型参数初始值
np.random.seed(612)
#w=tf.Variable(np.random.randn())
#b=tf.Variable(np.random.randn())
w=np.random.randn()
b=np.random.randn()

mse_train=[]
#mse_test=[]
for i in range(0,iter+1):
    #with tf.GradientTape() as tape:
     #   pred_train=w*x_train+b
      #  loss_train=0.5*tf.reduce_mean(tf.square(y_train-pred_train))
       # pred_test=w*x_test+b
        #loss_test=0.5*tf.reduce_mean(tf.square(y_test-pred_test))
    #mse_train.append(loss_train)
    #mse_test.append(loss_test)
    dL_dw=np.mean(x_train*(w*x_train+b-y_train))
    dL_db=np.mean(w*x_train+b-y_train)
    w=w-learn_rate*dL_dw
    b=b-learn_rate*dL_db
    pred=w*x_train+b
    Loss=np.mean(np.square(y_train-pred))/2
    mse_train.append(Loss)
    #dL_dW,dL_dB=tape.gradient(loss_train,[w,b])
    #w.assign_sub(learn_rate*dL_dW)
    #b.assign_sub(learn_rate*dL_dB)

    if i % display_step == 0:
        print("i:%i,Loss:%f,w:%f,b:%f"%(i,mse_train[i],w,b))

        #print("i:%i,Train Loss:%f,Test Loss:%f"%(i,loss_train,loss_test))
#print("w=",w)
#print("b=",b)