# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pyecharts.options as opts
from pyecharts.charts import Line
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess=tf.compat.v1.Session(config=config)

class CheckpointPB(tf.keras.callbacks.Callback):

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(CheckpointPB, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                            save_tflite(self.model)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


def train(model,
         loss_func,
         train_batch_gen,
         valid_batch_gen,
         learning_rate = 1e-4,
         nb_epoch = 300,
         saved_weights_name = 'best_weights.h5',
         class_num=1,
         anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
         w_name="",
         alpha=0.75,
         lable=""
         ):
    """A function that performs training on a general keras model.

    # Args
        model : tensorflow.keras.models.Model instance
        loss_func : function
            refer to https://keras.io/losses/

        train_batch_gen : tensorflow.keras.utils.Sequence instance
        valid_batch_gen : tensorflow.keras.utils.Sequence instance
        learning_rate : float
        saved_weights_name : str
    """
    # 1. create optimizer
    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    # 2. create loss function
    model.compile(loss=loss_func,
                  optimizer=optimizer)
    
    # 4. training
    train_start = time.time()
    try:
        history=model.fit_generator(generator = train_batch_gen,
                        steps_per_epoch  = len(train_batch_gen), 
                        epochs           = nb_epoch,
                        validation_data  = valid_batch_gen,
                        validation_steps = len(valid_batch_gen),
                        callbacks        = _create_callbacks(saved_weights_name),                        
                        verbose          = 1,
                        workers          = 3,
                        max_queue_size   = 8)
        print(history.history)
        
        plt.figure("loss")
        plt.grid()
        num1=1
        num2=0
        num3=3
        num4=4
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('loss')
        plt.legend(['train', 'test'], loc='upper left')
        plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
        i="./Model_file"+"/yolov2_object_recognition_"+w_name+"_"+str(alpha)+"-"+time_()
        os.mkdir(i)

        plt.savefig(i+'/Loss.jpg')
        loss_=[]
        loss_val=[]
        for o in history.history['loss']:
            loss_.append(round(o,4))
        for o in history.history['val_loss']:
            loss_val.append(round(o,4))
        c = (
            Line()
            .add_xaxis(range(1,len(loss_)+1))
            .add_yaxis("Train",loss_, is_smooth=True,linestyle_opts=opts.LineStyleOpts(width=3),is_symbol_show=False,color="#2196F3")
            .add_yaxis("Test", loss_val, is_smooth=True,linestyle_opts=opts.LineStyleOpts(width=3),is_symbol_show=False,color="#F9A825")
            .set_global_opts(title_opts=opts.TitleOpts(title="Loss rate"),
                            toolbox_opts=opts.ToolboxOpts(is_show=True,orient="vertical",pos_left="right",feature=opts.ToolBoxFeatureOpts(save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(background_color="#fff"),
                                                                                                                                            magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False),
                                                                                                                                            data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=False),
                                                                                                                                            data_view=opts.ToolBoxFeatureSaveAsImageOpts(is_show=False),
                                                                                                                                            brush=opts.ToolBoxFeatureBrushOpts(type_='rect')),
                                                                                                                                            ),
                            datazoom_opts=opts.DataZoomOpts(is_show=True,range_end=100,range_start=0,filter_mode="none"),
                            tooltip_opts=opts.TooltipOpts(is_show=True),
                            legend_opts=opts.LegendOpts(legend_icon="circle"),
                            axispointer_opts=opts.AxisPointerOpts(is_show=True))
            .set_series_opts(splitline_opts=opts.SplitLineOpts(is_show=True))
            .render(i+"/Loss.html")
        )

    except KeyboardInterrupt:
        save_tflite(model,class_num,w_name,alpha,anchors,lable,i)
        raise

    _print_time(time.time() - train_start)
    save_tflite(model,class_num,w_name,alpha,anchors,lable,i)

def _print_time(process_time):
    if process_time < 60:
        print("{:d}-seconds to train".format(int(process_time)))
    else:
        print("{:d}-mins to train".format(int(process_time/60)))
def time_():
    now_time=time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time()))
    return now_time
def save_tflite(model,num,w_name,alpha,anchors,label,i):
    ## waiting for kpu to support V4 - nncase >= 0.2.0
    ## https://github.com/kendryte/nncase
    #converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #tflite_model = converter.convert()
    
    ## kpu V3 - nncase = 0.1.0rc5
    output_layer="detection_layer_"+str(num)+"/BiasAdd"
    model.save(i+"/yolov2.h5", include_optimizer=False)
    tf.compat.v1.disable_eager_execution()
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(i+"/yolov2.h5",
                                        output_arrays=[output_layer])
    
    tfmodel = converter.convert()
    file = open (i+"/yolov2.tflite" , "wb")
    file.write(tfmodel)
    anchorstxt=open(i+"/anchors.txt","w")
    anchorstxt.write(str(anchors).replace("[","").replace("]",""))
    anchorstxt.close()
    lable=open(i+"/lable.txt","w")
    lable.write(str(label).replace("[","").replace("]","").replace("'",""))
    lable.close()
    # os.startfile(os.getcwd()+"/"+i)

def _create_callbacks(saved_weights_name):
    # Make a few callbacks
    early_stop = EarlyStopping(monitor='val_loss', 
                       min_delta=0.001, 
                       patience=20, 
                       mode='min', 
                       verbose=1,
                       restore_best_weights=True)
    checkpoint = CheckpointPB(saved_weights_name, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min', 
                                 period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001, verbose=1)
    callbacks = [early_stop, reduce_lr]
    return callbacks
