import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Dropout, LeakyReLU, Flatten
from keras.models import Model
from keras import optimizers

#%matplotlib inline
from matplotlib import pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# specify a seed for repeating the exact dataset splits
np.random.seed(seed=28213)

input_name = 'data/yeast_genotype_train.txt'
df_ori = pd.read_csv(input_name, sep='\t', index_col=0)
print(df_ori.shape)

# one hot encode
df_onehot = to_categorical(df_ori)
df_onehot.shape

# split df to train and valid
train_X, valid_X = train_test_split(df_onehot, test_size=0.2)

train_X.shape, valid_X.shape

# hyperparameters
missing_perc = 0.1

# training
batch_size = 32
lr = 1e-3
epochs = 10

latent_dim = 32
height = 32
width = 32
channels = 3

feature_size = train_X.shape[1]
inChannel = train_X.shape[2]

drop_prec = 0.25

generator_input = Input(shape=(feature_size, inChannel))
#
# 首先，将输入转换为16x16 128通道的feature map
x = Conv1D(32, 5, padding='same')(generator_input)
x = LeakyReLU()(x)
x = Dropout(drop_prec)(x)

# 然后，添加卷积层
x = Conv1D(64, 5, padding='same')(x)
x = LeakyReLU()(x)
x = Dropout(drop_prec)(x)

x = Conv1D(128, 5, padding='same')(x)
x = LeakyReLU()(x)
x = Dropout(drop_prec)(x)

x = Conv1D(64, 5, padding='same')(x)
x = LeakyReLU()(x)
x = Dropout(drop_prec)(x)

x = Conv1D(32, 5, padding='same')(x)
x = LeakyReLU()(x)
x = Dropout(drop_prec)(x)

x = Conv1D(3, 5, activation='softmax', padding='same', name='gen_output')(x)

generator = Model(generator_input, x)
print("Generator:")
generator.summary()

generator_optimizer = optimizers.Adam(lr=8e-5, clipvalue=1.0, decay=1e-8)
generator.compile(optimizer=generator_optimizer, loss='categorical_crossentropy')

'''
discriminator(鉴别器)
创建鉴别器模型，它将候选图像（真实的或合成的）作为输入，并将其分为两类：“生成的图像”或“来自训练集的真实图像”。
'''
discriminator_input = Input(shape=(feature_size, inChannel))
y = Conv1D(32, 5)(discriminator_input)
y = LeakyReLU()(y)
y = Conv1D(64, 5, strides=2)(y)
y = LeakyReLU()(y)
y = Conv1D(32, 5, strides=2)(y)
y = LeakyReLU()(y)

y = Flatten()(y)

y = Dense(512)(y)
y = LeakyReLU()(y)
# 重要的技巧（添加一个dropout层）
##################################################################
#################################################################
y = Dropout(0.25)(y)

# 分类层
y = Dense(1, activation='sigmoid')(y)

discriminator = Model(discriminator_input, y)
discriminator.summary()

# 为了训练稳定，在优化器中使用学习率衰减和梯度限幅（按值）。
discriminator_optimizer = optimizers.SGD(lr=8e-5, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

'''
The adversarial network:对抗网络
最后，设置GAN，它链接生成器（generator）和鉴别器（discrimitor）。 这是一种模型，经过训练，
将使生成器（generator）朝着提高其愚弄鉴别器（discrimitor）能力的方向移动。 该模型将潜在的空间点转换为分类决策，
“假的”或“真实的”，并且意味着使用始终是“这些是真实图像”的标签来训练。 所以训练`gan`将以一种方式更新
“发生器”的权重，使得“鉴别器”在查看假图像时更可能预测“真实”。 非常重要的是，将鉴别器设置为在训练
期间被冻结（不可训练）：训练“gan”时其权重不会更新。 如果在此过程中可以更新鉴别器权重，那么将训练鉴别
器始终预测“真实”。
'''
# 将鉴别器（discrimitor）权重设置为不可训练（仅适用于`gan`模型）
discriminator.trainable = False

gan_input = Input(shape=(feature_size, inChannel))
#gen_output = generator(gan_input)
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
#gan = Model(inputs=[generator_input, gan_input], outputs=[gen_output, gan_output])
gan_optimizer = optimizers.Adam(lr=4e-5, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

#gan.compile(optimizer=gan_optimizer,
            #loss={'gen_output': 'categorical_crossentropy', 'gan_output': 'binary_crossentropy'})

'''
  开始训练了。
  每个epoch：
   *在潜在空间中绘制随机点（随机噪声）。
   *使用此随机噪声生成带有“generator”的图像。
   *将生成的图像与实际图像混合。
   *使用这些混合图像训练“鉴别器”，使用相应的目标，“真实”（对于真实图像）或“假”（对于生成的图像）。
   *在潜在空间中绘制新的随机点。
   *使用这些随机向量训练“gan”，目标都是“这些是真实的图像”。 这将更新发生器的权重（仅因为鉴别器在“gan”内被冻结）
   以使它们朝向获得鉴别器以预测所生成图像的“这些是真实图像”，即这训练发生器欺骗鉴别器。
'''
def cal_prob(predict_missing_onehot):
    # calcaulate the probility of genotype 0, 1, 2
    predict_prob = predict_missing_onehot[:,:,1:3] / predict_missing_onehot[:,:,1:3].sum(axis=2, keepdims=True)
    return predict_prob[0]

iterations = 500
batch_size = 32
save_dir = '.\\gan_image'

#Generate data
import copy
#Missing_train = train_X.copy()
Missing_train = copy.deepcopy(train_X)
for i in range(Missing_train.shape[0]):
    missing_size = int(missing_perc * Missing_train.shape[1])
    missing_index = np.random.randint(Missing_train.shape[1], size=missing_size)
            # missing loci are encoded as [0, 0]
    Missing_train[i, missing_index, :] = [1, 0, 0]

#Missing_valid = valid_X.copy()
Missing_valid = copy.deepcopy(valid_X)

# 开始训练迭代
for step in range(iterations):
    # 在潜在空间中抽样随机点
    start = 0

    state = np.random.get_state()
    np.random.shuffle(Missing_train)

    np.random.set_state(state)
    np.random.shuffle(train_X)

    while start < len(train_X):
        #random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
        stop = start + batch_size
        if(stop > len(train_X)):
            break
        # 将随机抽样点解码为假图像
        #print("Missing_train[start:stop].shape:", Missing_train[start:stop].shape)
        generated_images = generator.predict(Missing_train[start:stop])
        #print("generated_images.shape:", generated_images.shape)
        # 将假图像与真实图像进行比较

        real_images = train_X[start: stop]
        combined_images = np.concatenate([generated_images, real_images])
        #print("combined_images.shape:", combined_images.shape)
        # 组装区别真假图像的标签
        labels = np.concatenate([np.zeros((batch_size, 1)),
                                 np.ones((batch_size, 1))])
        # 重要的技巧，在标签上添加随机噪声
        #labels += 0.05 * np.random.random(labels.shape)

        # 训练鉴别器（discrimitor）
        d_fake_loss = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
        d_real_loss = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        # d_fake_loss = discriminator.train_on_batch(generated_images, labels[0:batch_size])
        # d_real_loss = discriminator.train_on_batch(real_images, labels[batch_size:64])

        # 在潜在空间中采样随机点
        #random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        # 汇集标有“所有真实图像”的标签
        misleading_targets = np.ones((batch_size, 1))

        # 训练生成器（generator）（通过gan模型，鉴别器（discrimitor）权值被冻结）
        gen_loss = generator.train_on_batch(Missing_train[start:stop], real_images)
        a_loss = gan.train_on_batch(Missing_train[start:stop], misleading_targets)
        #gen_loss, a_loss = gan.train_on_batch([Missing_train[start:stop], Missing_train[start:stop]],[real_images, misleading_targets])

        print('discriminator fake loss at step %s: %s' % (step, d_fake_loss))
        print('discriminator real loss at step %s: %s' % (step, d_real_loss))
        print('generator pixel loss at step %s: %s' % (step, gen_loss))
        print('adversarial loss at step %s: %s' % (step, a_loss))

        start += batch_size

    if step % 1 == 0:
        # 保存网络权值
        #gan.save_weights('gan.h5')

        # 输出metrics

        avg_accuracy = []
        for i in range(Missing_valid.shape[0]):
            # predict

            missing_size = int(missing_perc * Missing_valid.shape[1])
            missing_val_index = np.random.randint(Missing_valid.shape[1],
                                                  size=missing_size)
            Missing_valid[i, missing_val_index, :] = [1, 0, 0]

            predict_onehot = generator.predict(Missing_valid[i:i + 1, :, :])
            # only care the missing position
            predict_missing_onehot = predict_onehot[0:1, missing_val_index, :]

            # calculate probability and save file.
            predict_prob = cal_prob(predict_missing_onehot)
            #     pd.DataFrame(predict_prob).to_csv('results/{}.csv'.format(df_ori.index[i]),
            #                                       header=[1, 2],
            #                                       index=False)

            # predict label
            predict_missing = np.argmax(predict_missing_onehot, axis=2)
            # real label
            label_missing_onehot = valid_X[i:i + 1, missing_val_index, :]
            label_missing = np.argmax(label_missing_onehot, axis=2)
            # accuracy
            correct_prediction = np.equal(predict_missing, label_missing)
            accuracy = np.mean(correct_prediction)
            if i % 200 == 0:
                print('{}/{}, sample ID: {}, accuracy: {:.4f}'.format(
                    i, Missing_valid.shape[0], df_ori.index[i], accuracy))

            avg_accuracy.append(accuracy)

        print('=======================The average imputation accuracy' \
              'on test data with {} missing genotypes is {:.4f}: ========================'
              .format(missing_perc, np.mean(avg_accuracy)))
        # # 保存生成的图像
        # img = image.array_to_img(generated_images[0] * 255., scale=False)
        # img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))
        #
        # # 保存真实图像，以便进行比较
        # img = image.array_to_img(real_images[0] * 255., scale=False)
        # img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))

