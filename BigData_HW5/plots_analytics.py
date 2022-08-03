import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# Make plots for loss curves and accuracy curves.
	# do not have to return the plots.
	# can save plots as files by codes here or an interactive way according to your preference.

    plt.plot(train_losses, label = 'Training Loss')
    plt.plot(valid_losses, label = 'Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend(loc = 'upper right')
    plt.savefig('Loss.png')
    plt.cla()
    # plt.show()

    plt.plot(train_accuracies, label = 'Training Accuracy')
    plt.plot(valid_accuracies, label = 'Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend(loc='upper right')
    plt.savefig('Accuracy.png')
    plt.cla()
    # plt.show()
    pass


def plot_confusion_matrix(results, class_names):
    # TODO: Make a confusion matrix plot.
	# do not have to return the plots.
	# can save plots as files by codes here or an interactive way according to your preference.

    ConMat = confusion_matrix(np.array(results)[:,0], np.array(results)[:,1])
    ConMat = ConMat.astype('float')/ConMat.sum(axis=1)[:, np.newaxis]

    plt.imshow(ConMat, interpolation = 'nearest', cmap = plt.cm.Blues)
    np.set_printoptions(precision=2)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    setPoint = ConMat.max() / 2.
    for m, n in itertools.product(range(ConMat.shape[0]), range(ConMat.shape[1])):
        plt.text(n, m, format(ConMat[m, n], '.2f'), horizontalalignment ='center', color='white' if ConMat[m,n] > setPoint else 'black')

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig("Confusion.png")
    # plt.show()

    pass
