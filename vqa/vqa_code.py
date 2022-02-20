from asyncio.windows_events import NULL
from importlib import import_module
from sysconfig import get_python_version
from black import gen_python_files


def vqa_func(video_path):
    import cv2
    import os, random
    from PIL import Image
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.compat.v1.keras.backend import set_session
    import sys, time, os, warnings 
    import numpy as np
    import pandas as pd 
    from collections import Counter 
    os.environ['SPACY_WARNING_IGNORE'] = 'W008'
    warnings.filterwarnings("ignore")
    print("python {}".format(sys.version))
    print("keras version {}".format(keras.__version__)); del keras
    print("tensorflow version {}".format(tf.__version__))
    from keras.preprocessing.image import load_img, img_to_array
    from keras.applications.vgg16 import preprocess_input
    from IPython.display import display
    from PIL import Image

    import keras
    from keras.applications.vgg16 import VGG16
    modelvgg = VGG16(include_top=True,weights=None)
    ## load the locally saved weights 
    modelvgg.load_weights("C:/Users/Rasika/Django/VQA_11/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
    # modelvgg.summary()
    from keras import models
    modelvgg.layers.pop() #the last layer is for classification so we remove that layer for feature extraction model
    modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)
    ## show the deep learning model
    # modelvgg.summary()




    from keras import layers
    from keras.layers import Input, Flatten, Dropout, Activation
    from keras.layers.advanced_activations import LeakyReLU, PReLU

    ## image feature

    dim_embedding = 64
    vocab_size = 4373
    input_image = layers.Input(shape=(1000,))
    fimage = layers.Dense(256,activation='relu',name="ImageFeature")(input_image)
    ## sequence model
    input_txt = layers.Input(shape=(29,))
    ftxt = layers.Embedding(vocab_size,dim_embedding, mask_zero=True)(input_txt)
    ftxt = layers.LSTM(256,name="CaptionFeature",return_sequences=True)(ftxt)
    #,return_sequences=True
    #,activation='relu'
    se2 = Dropout(0.04)(ftxt)
    ftxt = layers.LSTM(256,name="CaptionFeature2")(se2)
    ## combined model for decoder
    decoder = layers.add([ftxt,fimage])
    decoder = layers.Dense(256,activation='relu')(decoder)
    output = layers.Dense(vocab_size,activation='softmax')(decoder)
    model = models.Model(inputs=[input_image, input_txt],outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # print(model.summary())




    model.load_weights("C:/Users/Rasika/Django/VQA_11/model50.h5")
    model.summary()

    import json
    with open("C:/Users/Rasika/Django/VQA_11/data.json") as json_file:
        index_word = json.load(json_file)
    
        # Print the type of data variable
        print("Type:", type(index_word))




    data= pd.read_csv("C:/Users/Rasika/Django/VQA_11/csvfile.csv")
    data = data.set_index('filename')




    # df_txt0.drop_duplicates(keep=False)
    data = data[~data.index.duplicated(keep='first')]




    data.reset_index(level=0, inplace=True)
    data[:5]




    dcaptions = data["caption"].values


    from keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from keras.preprocessing.text import Tokenizer
    ## the maximum number of words in dictionary
    nb_words = 6000
    tokenizer = Tokenizer(nb_words=nb_words)
    tokenizer.fit_on_texts(dcaptions)
    vocab_size = len(tokenizer.word_index) + 1
    def predict_caption(image):
        '''
        image.shape = (1,4462)
        '''

        in_text = 'startseq'
        maxlen = 29
        for iword in range(maxlen):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence],maxlen)
            yhat = model.predict([image,sequence],verbose=0)
            yhat = np.argmax(yhat)
            
            newword = index_word[str(yhat)]
            in_text += " " + newword
            if newword == "endseq":
                break
        return(in_text)




    # path = "../input/videos"
    # video = random.choice(os.listdir(path))
    # video_path = path + "/" + video
    # print(video_path)
    # Read the video from specified path
    cam = cv2.VideoCapture(video_path)
    
    try:
        
        # creating a folder named data
        if not os.path.exists('data2'):
            os.makedirs('data2')
    
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
    
    # frame
    currentframe = 0
    namelist = []
    
    while(True):
        
        # reading from frame
        ret,frame = cam.read()
    
        if ret:
            # if video is still left continue creating images
            name = './data2/frame' + str(currentframe) + '.jpg'
            # print ('Creating...' + name)
            namelist.append(name)
    
            # writing the extracted images
            cv2.imwrite(name, frame)
    
            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()




    import random
    namelist_sorted = sorted(list(random.sample(namelist, 10)))
    print(namelist_sorted)




    def generate_description(filename):
        npix = 224 #image size is fixed at 224 because VGG16 model has been pre-trained to take that size.
        target_size = (npix,npix,3)
        image_load = load_img(filename, target_size=target_size)

        plt.imshow(image_load)


        image = load_img(filename, target_size=target_size)
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        nimage = preprocess_input(image)

        y_pred = modelvgg.predict(nimage.reshape( (1,) + nimage.shape[:3]))
        image_feature = y_pred.flatten()
        # print(image_feature.shape)
        caption = predict_caption(image_feature.reshape(1,len(image_feature)))
        return caption




    des = ""
    for name in namelist_sorted:
        des += generate_description(name)[9:-7] + ", "
    print(des)

    # gen_python_files().system('pip install git+https://github.com/ramsrigouthamg/Questgen.ai')
    # get_python_version().system('pip install git+https://github.com/boudinfl/pke.git')

    # get_ipython().system('python -m nltk.downloader universal_tagset')
    # get_ipython().system('python -m spacy download en ')
    # get_ipython().system('wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz')
    # get_ipython().system('tar -xvf  s2v_reddit_2015_md.tar.gz')
    # get_ipython().system('pip install --quiet git+https://github.com/boudinfl/pke.git@dc4d5f21e0ffe64c4df93c46146d29d1c522476b')
    # get_ipython().system('pip install --quiet flashtext==2.7')

    from pprint import pprint
    # import nltk
    # nltk.download('stopwords')
    import sys
    sys.path.insert(1, 'C:/Users/Rasika/Django/VQA_11/Questgen.ai-master/Questgen')
    import main


    def get_mcq(payload):
        qg = main.QGen()
        output = qg.predict_mcq(payload)
        return output
    
    payload = {
                    "input_text": des
        }

    # getting MCQs from the text
    import random


    output = get_mcq(payload)
    final_list = {}
    i=1

    # if 'questions' in output:
    #     del output['questions']
    if 'questions' in output:

        for j in range(0,len(output['questions'])):
            choices = output['questions'][j]['options']
            an = random.randint(0,len(choices))
            choices.insert(an, output['questions'][j]['answer'])
            final_list[i] = {'Question': output['questions'][j]['question_statement'], 'Choices': choices, 'Answer': output['questions'][j]['answer']}
            i+=1

        # print(output)
        # print(final_list)

        return final_list
    else:
        return NULL

