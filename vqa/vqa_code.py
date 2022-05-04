import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import os, sys
import pickle, functools, operator
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import joblib
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import random
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import datetime
from collections import Counter
from pprint import pprint
# import nltk
# nltk.download('stopwords')
import sys
import shutil

import tqdm
import numpy as np
import cv2
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

# class to perform inference on all test files and save as test_output.txt
class Video2Text(object):
        ''' Initialize the parameters for the model '''
        def __init__(self):
            self.latent_dim = 512
            self.num_encoder_tokens = 4096
            self.num_decoder_tokens = 1500
            self.time_steps_encoder = 80
            self.time_steps_decoder = None
            self.preload = True
            self.preload_data_path = 'preload_data'
            self.max_probability = -1

            # processed data
            self.encoder_input_data = []
            self.decoder_input_data = []
            self.decoder_target_data = []
            self.tokenizer = None

            # models
            self.encoder_model = None
            self.decoder_model = None
            self.inf_encoder_model = None
            self.inf_decoder_model = None
            # self.save_model_path = '../input/model-final'
        
        def load_inference_models(self):
            # load tokenizer
            
            with open('C:/Users/Rasika/Django/VQA_11/tokenizer1500', 'rb') as file:
                self.tokenizer = joblib.load(file)

            # inference encoder model
            self.inf_encoder_model = load_model('C:/Users/Rasika/Django/VQA_11/encoder_model.h5')

            # inference decoder model
            decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
            decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
            decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
            decoder_state_input_h = Input(shape=(self.latent_dim,))
            decoder_state_input_c = Input(shape=(self.latent_dim,))
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
            decoder_states = [state_h, state_c]
            decoder_outputs = decoder_dense(decoder_outputs)
            self.inf_decoder_model = Model(
                [decoder_inputs] + decoder_states_inputs,
                [decoder_outputs] + decoder_states)
            self.inf_decoder_model.load_weights('C:/Users/Rasika/Django/VQA_11/decoder_model_weights.h5')
        
        def decode_sequence2bs(self, input_seq):
            states_value = self.inf_encoder_model.predict(input_seq)
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, self.tokenizer.word_index['startseq']] = 1
            self.beam_search(target_seq, states_value,[],[],0)
            return decode_seq

        def beam_search(self, target_seq, states_value, prob,  path, lens):
            global decode_seq
            node = 2
            output_tokens, h, c = self.inf_decoder_model.predict(
                [target_seq] + states_value)
            output_tokens = output_tokens.reshape((self.num_decoder_tokens))
            sampled_token_index = output_tokens.argsort()[-node:][::-1]
            states_value = [h, c]
            for i in range(node):
                if sampled_token_index[i] == 0:
                    sampled_char = ''
                else:
                    sampled_char = list(self.tokenizer.word_index.keys())[list(self.tokenizer.word_index.values()).index(sampled_token_index[i])]
                MAX_LEN = 10
                if(sampled_char != 'endseq' and lens <= MAX_LEN):
                    p = output_tokens[sampled_token_index[i]]
                    if(sampled_char == ''):
                        p = 1
                    prob_new = list(prob)
                    prob_new.append(p)
                    path_new = list(path)
                    path_new.append(sampled_char)
                    target_seq = np.zeros((1, 1, self.num_decoder_tokens))
                    target_seq[0, 0, sampled_token_index[i]] = 1.
                    self.beam_search(target_seq, states_value, prob_new, path_new, lens+1)
                else:
                    p = output_tokens[sampled_token_index[i]]
                    prob_new = list(prob)
                    prob_new.append(p)
                    p = functools.reduce(operator.mul, prob_new, 1)
                    if(p > self.max_probability):
                        decode_seq = path
                        self.max_probability = p

        def decoded_sentence_tuning(self, decoded_sentence):
            decode_str = []
            filter_string = ['startseq', 'endseq']
            unigram = {}
            last_string = ""
            for idx2, c in enumerate(decoded_sentence):
                if c in unigram:
                    unigram[c] += 1
                else:
                    unigram[c] = 1
                if(last_string == c and idx2 > 0):
                    continue
                if c in filter_string:
                    continue
                if len(c) > 0:
                    decode_str.append(c)
                if idx2 > 0:
                    last_string = c
            return decode_str

        def get_test_data(self, video_name, path):
            X_test = []
            X_test_filename = []
        
            for i in path:
                filename = i.split('.')[0]
                f = np.load(os.path.join('C:/Users/Rasika/Django/VQA_11/data/features_dir', video_name + '.npy'))
                X_test.append(f)
                X_test_filename.append(filename[:-4])
            X_test = np.array(X_test)
            return X_test, X_test_filename

        def test(self, video_name,a):
            X_test, X_test_filename = self.get_test_data(video_name, a)
            print(len(X_test), len(X_test_filename))
            # generate inference test outputs

            for idx, x in enumerate(X_test): 
                decoded_sentence = self.decode_sequence2bs(x.reshape(-1, 80, 4096))
                decode_str = self.decoded_sentence_tuning(decoded_sentence)
                sent=''
                for d in decode_str:
                    sent+=d + ' '
                print(sent)
                # re-init max prob
                self.max_probability = -1
            return sent

# Main function to be called for processing      
def vqa_func(video_name, video_path):

    # Convert video to frames saved in a temporary folder with 80frames/video
    def video_to_frames(video):

        if os.path.exists('temporary_images'):
            shutil.rmtree('temporary_images')
        os.makedirs('temporary_images')
        # video_path = os.path.join('../input/msvd-dataset/YouTubeClips/YouTubeClips', video)
        
        count = 0
        image_list = []
        # Path to video file
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            cv2.imwrite(os.path.join('./temporary_images', 'frame%d.jpg' % count), frame)
            image_list.append(os.path.join('./temporary_images', 'frame%d.jpg' % count))
            count += 1

        cap.release()


        return image_list

    def model_cnn_load():
        model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
        out = model.layers[-2].output
        model_final = Model(inputs=model.input, outputs=out)
        return model_final


    def load_image(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        return img


    # Image names list will be loaded and passed to CNN model, features will be returned
    def extract_features(video, model):
        """
        :param video: The video whose frames are to be extracted to convert into a numpy array
        :param model: the pretrained vgg16 model
        :return: numpy array of size 4096x80
        """
        video_id = video.split(".")[0]
        print(video_id)
        print(f'Processing video {video}')

        image_list = video_to_frames(video)
        samples = np.round(np.linspace(0, len(image_list) - 1, 80))
        image_list = [image_list[int(sample)] for sample in samples]
        images = np.zeros((len(image_list), 224, 224, 3))
        for i in range(len(image_list)):
            img = load_image(image_list[i])
            images[i] = img
        images = np.array(images)
        fc_feats = model.predict(images, batch_size=3)
        img_feats = np.array(fc_feats)
        # cleanup
        return img_feats


    # Features will be saved in .npy file
    def extract_feats_pretrained_cnn(video_name, video):
        """
        saves the numpy features from all the videos
        """
        model = model_cnn_load()
        print('Model loaded')

        if not os.path.exists('data/features_dir'):
            os.makedirs('data/features_dir')
        txt=video.split("/")[-1]

        t=str(txt).strip()


        outfile = os.path.join('C:/Users/Rasika/Django/VQA_11/data', 'features_dir', video_name + '.npy')
        img_feats = extract_features(txt, model)
        np.save(outfile, img_feats)
        validation_set=[]
        validation_set.append(txt)

        return validation_set

    a=extract_feats_pretrained_cnn(video_name, video_path)
    c = Video2Text()
    c.load_inference_models()
    caption_true=c.test(video_name, a)
    print(caption_true)

    sys.path.insert(1, 'C:/Users/Rasika/Django/VQA_11/Questgen.ai-master/Questgen')
    import main


    def get_mcq(payload):
        qg = main.QGen()
        output = qg.predict_mcq(payload)
        return output
    
    payload = {
                    "input_text": caption_true
        }

    # getting MCQs from the text
    import random


    output = get_mcq(payload)
    final_list = {}
    i=1

    print(output)

    # if 'questions' in output:
    #     del output['questions']
    if 'questions' in output:

        for j in range(0,len(output['questions'])):
            choices = output['questions'][j]['options']
            an = random.randint(0,len(choices))
            choices.insert(an, output['questions'][j]['answer'])
            final_list[i] = {'Question': output['questions'][j]['question_statement'], 'Choices': choices, 'Answer': output['questions'][j]['answer']}
            i+=1

        return final_list
    else:
        return None
