import matplotlib.pyplot as plt
from keras import backend as K


class ModelVisualizer:

    def __init__(self, model, dh):
        self.model = model
        self.dh = dh

    def visualize_gru(self, matId, eval):
        _type = 'Adapt'
        if eval:
            _type = 'Eval'
        orig = self.dh.conf_dict_mat[matId][0].reshape(self.dh.matN[matId], self.dh.matM[matId])
        plt.imshow(orig, interpolation="nearest", cmap="gray")
        plt.savefig('./figures/' + _type + '_' + str(matId) + '_orig_GRU.jpg', bbox_inches='tight', format='jpg')
        cn = self.model.layers[0].get_weights()[9].reshape(3, 32)
        plt.imshow(cn, interpolation="nearest", cmap="gray")
        plt.savefig('./figures/' + _type + '_' + str(matId) + '_filter_GRU.jpg', bbox_inches='tight', format='jpg')
        gruout1_f = K.function([self.model.layers[8].input], [self.model.layers[9].output])
        if eval:
            new_mat = gruout1_f([self.dh.conf_dict_seq[matId]])[0].reshape(self.dh.matN[matId], self.dh.matM[matId])
        else:
            new_mat = gruout1_f([self.dh.conf_dict_seq[matId]])[0].reshape(self.dh.matN[matId], self.dh.matM[matId])
        plt.imshow(new_mat, interpolation="nearest", cmap="gray")
        plt.savefig('./figures/' + _type + '_' + str(matId) + '_filter_applied_GRU.jpg', bbox_inches='tight', format='jpg')
        if not eval:
            adapted = self.model.predict_classes(self.dh.conf_dict_seq[matId], verbose=2)
            adapted = adapted[0].reshape(self.dh.matN[matId], self.dh.matM[matId])
            plt.imshow(adapted, interpolation="nearest", cmap="gray")
            plt.savefig('./figures/' + _type + '_' + str(matId) + '_adapted_GRU.jpg', bbox_inches='tight', format='jpg')

    def visualize_cnn(self, matId, eval):
        _type = 'Adapt'
        if eval:
            _type = 'Eval'
        orig = self.dh.conf_dict_mat[matId][0].reshape(self.dh.matN[matId], self.dh.matM[matId])
        plt.imshow(orig, interpolation="nearest", cmap="gray")
        plt.savefig('./figures/' + _type + '_' + str(matId) + '_orig_CNN.jpg', bbox_inches='tight', format='jpg')
        convout1_f = K.function([self.model.layers[0].input], [self.model.layers[2].output])
        new_mat = convout1_f([self.dh.conf_dict_mat[matId]])
        cn = self.model.layers[1].get_weights()[0][0][1]
        plt.imshow(cn, interpolation="nearest", cmap="gray")
        plt.savefig('./figures/' + _type + '_' + str(matId) + '_filter_CNN.jpg', bbox_inches='tight', format='jpg')
        plt.imshow(new_mat[0][0][1], interpolation="nearest", cmap="gray")
        plt.savefig('./figures/' + _type + '_' + str(matId) + '_filter_applied_CNN.jpg', bbox_inches='tight', format='jpg')

