"""
    Original: https://github.com/sdierauf/tiny-imagenet-classifier/blob/master/plotter.py
    Modified: Eric Masseran from 29th May
"""
from plotting_utils import PlotGenerator
from keras.callbacks import History

class Plotter(History):
    # see PlotGenerator.__init__() for a description of the parameters
    def __init__(self,
                 save_to_filepath=None, show_plot_window=False,
                 linestyles=None, linestyles_first_epoch=None,
                 show_regressions=True,
                 poly_forward_perc=0.1, poly_backward_perc=0.2,
                 poly_n_forward_min=5, poly_n_backward_min=10,
                 poly_degree=1):
        super(Plotter, self).__init__()
        pgen = PlotGenerator(linestyles=linestyles,
                             linestyles_first_epoch=linestyles_first_epoch,
                             show_regressions=show_regressions,
                             poly_forward_perc=poly_forward_perc,
                             poly_backward_perc=poly_backward_perc,
                             poly_n_forward_min=poly_n_forward_min,
                             poly_n_backward_min=poly_n_backward_min,
                             poly_degree=poly_degree,
                             show_plot_window=show_plot_window,
                             save_to_filepath=save_to_filepath)
        self.plot_generator = pgen

    def on_epoch_end(self, epoch, logs={}):
        super(Plotter, self).on_epoch_end(epoch, logs)
        dv = self.params['do_validation']
        # print()
        # print(self.params)
        # print(vars(self))
        #sa = self.history['acc']

        train_loss = self.history['loss']
        val_loss = self.history['val_loss'] if dv else []
        #train_acc = self.history['acc']if sa else []
        #val_acc = self.history['val_acc'] if dv and sa else []

        self.plot_generator.update(epoch, train_loss, val_loss)
