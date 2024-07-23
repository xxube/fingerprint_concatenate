#from Landrum 2014 paper


import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

def display_labelled_cmat(cmat, labels, label_extras=None, xlabel=True, ylabel=True, rotation=90):
   
    # Display a labeled confusion matrix.

    # Args:
        # cmat (array-like): Confusion matrix.
        # labels (list): List of labels for the matrix.
        # label_extras (dict, optional): Additional label information. Defaults to None.
        # xlabel (bool, optional): Whether to display x-axis labels. Defaults to True.
        # ylabel (bool, optional): Whether to display y-axis labels. Defaults to True.
        # rotation (int, optional): Rotation angle for x-axis labels. Defaults to 90.
  
    
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    pax = ax.pcolor(cmat, cmap=plt.get_cmap('Blues', 100000), vmin=0, vmax=1)
    ax.set_frame_on(True)

    print(np.sum(cmat))

    # Put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(cmat.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(cmat.shape[1]) + 0.5, minor=False)

    # Table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    if label_extras is not None:
        labels = [' %s %s' % (x, label_extras[x].strip()) for x in labels]
    
    ax.set_xticklabels([], minor=False) 
    ax.set_yticklabels([], minor=False)

    if xlabel:
        ax.set_xticklabels(labels, minor=False, rotation=rotation, horizontalalignment='left', size=14)
        plt.xlabel('True reaction label')
        ax.xaxis.set_label_position('top')
    if ylabel:
        ax.set_yticklabels(labels, minor=False, size=14)
        plt.ylabel('Predicted reaction label')

    fig.colorbar(pax)
    plt.axis('tight')
    plt.show()

if __name__ == '__main__':
    short_labels = ['NR', 'SN', 'Elim.', 'SN+m', 'E+m', 'H-X M.', 'H-X AntiM.', 
                    'H2O M.', 'H2O AntiM.', 'R-OH (M.)', 'H2', 'X-X Add.', 'X-OH Add.', 'Epox.', 
                    'Hydrox.', 'Ozon.', 'Polymer.']

    full_labels = ['Null Reaction', 'Nucleophilic substitution', 'Elimination', 
                   'Nucleophilic Substitution with Methyl Shift', 'Elimination with methyl shift', 
                   'Hydrohalogenation (Markovnikov)', 'Hydrohalogenation (Anti-Markovnikov)', 
                   'Hydration (Markovnikov)', 'Hydration (Anti-Markovnikov)', 'Alkoxymercuration-demercuration', 
                   'Hydrogenation', 'Halogenation', 'Halohydrin formation', 'Epoxidation', 
                   'Hydroxylation', 'Ozonolysis', 'Polymerization']

    num_labels = [str(i) for i in range(len(full_labels))]
    
    with open('opt/class_3_3_perbenzacid_morgan1_50eps_results.dat', 'rb') as file:
        results = pkl.load(file)
    
    cmat = results['confusion_matrix'][1]
    print(type(cmat))
    
    display_labelled_cmat(cmat, num_labels, rotation=0)
