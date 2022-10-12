import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from wordcloud import WordCloud

def plot_confusion_matrix(y_test, y_pred):
    """Plot a confusion matrix
    
    Args:
        y_test (pandas series, nparray): Target test set
        y_pred (pandas series, nparray): Predicted targets

    Returns:
        Figure containing the confusion matrix.
    """    

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()


def plot_roc_auc(fpr, tpr, auc):
    """Plot a ROC curve
    
    Args:
        fpr (pandas series, nparray): false positive rate.
        tpr (pandas series, nparray): true positive rate.
        auc (float) : auc model score.
        
    Returns:
        Figure containing the ROC curve and its AUC.
    """  
    
    plt.rcParams['figure.figsize'] = (12., 8.)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.legend(loc=4)


def plot_word_cloud(df, text_column, class_column=None, sentiment_class=None):
    """Plot word cloud

    Args:
        df (pandas dataframe): Data frame with documents.
        text_column (str): Data frame column with the texts.
        class_column (str, optional): Target column. Defaults to None.
        sentiment_class (str, optional): Target classes in target column, 
            could be "neg" or "pos". Defaults to None.
            
    Returns:
        Figure containing the word cloud
    """
    
    if sentiment_class:
        df = df[df[class_column] == sentiment_class]
        
    all_words = ' '.join([text for text in df[text_column]])

    # Building the word cloud plot with only single words (collocations=False)
    wordcloud_pt = WordCloud(width= 800, height= 500,
                      max_font_size = 110,
                      collocations = False).generate(all_words)

    # Plotting the figure, setting interpolation = "bilinear" for a better 
    # figure constrast
    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud_pt, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    