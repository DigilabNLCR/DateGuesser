"""
This python script serves to guess the date of the composition based on pre-trained models.
"""

__version__ = '0.1.0'
__author__  = 'František Válek'

""" This version is a beta version serving to illustrate the guessing for poet Adolf Heyduk. The results are, addmittedly, very poor so far. """


def get_composition_et_vectorizer_from_model_name(model_name:str):
    """
    Get the composition and vectorizer from the model name.

    :param model_name: The model name.
    """
    end_comp = model_name.find('_model')
    composition = model_name[4:end_comp]
    vectorizer = model_name.replace('_model', '_vectorizer')
    end_vect = vectorizer.find('.m-')
    vectorizer = vectorizer[:end_vect]+'.joblib'
    
    return composition, vectorizer


def guess_instance_segment(model_filename:str, vectorizer_filename:str, data_to_eval:str, models_path=MODELS_PATH):
    """
    This function is used within the guess_file function as a guess of one instance.
    
    :param model_filename: quite self-explanatory...
    :param data_to_eval: input data must be a str in XML-like structure, already delexicalized (with "<token>TOKEN</token>")
    """

    model = load(os.path.join(models_path, model_filename))
    vectorizer = load(os.path.join(models_path, vectorizer_filename))

    # Create test counts
    X_guess_counts = vectorizer.transform([data_to_eval])

    # Prediction and evaluation:
    y_guess_pred = model.predict(X_guess_counts)
    
    print(f'\tGUESS: model - {model_filename}, guessed time period - {y_guess_pred[0]}')

    return y_guess_pred[0]


def evaluate_model_on_loo_compostition(model_name:str, models_path=MODELS_PATH):
    """
    This function evaluates the model on the leave-one-out composition.

    :param model_name: The name of the model file (in joblib format).
    :param models_path: The path to the models.
    """
    composition, vectorizer = get_composition_et_vectorizer_from_model_name(model_name)

    print('Evaluating model on', composition)

    # TODO: implement the r-designations for the trainnig and testing datasets. Show this on the models filenames, etc. Also, show in the models filenames the window size, segment_window, and shift_size.
    data, targets = segment_file_to_data_et_targets(os.path.join(DATA_PATH_HEYDUK, composition), year_window_size=10, segment_window=10, shift_size=5, feature_type='lemma', autosem_delexicalise=True, ignore_interpunkt=True)

    Hejduk_test_dateship = NKPdateship(data, targets)

    score = {'correct': 0, 'incorrect': 0}
    detail = defaultdict(int)

    for i, test_data in enumerate(Hejduk_test_dateship.data):
        true_period = Hejduk_test_dateship.targets[i]

        guessed_period = guess_instance_segment(model_name, vectorizer, test_data, models_path=models_path)

        if guessed_period == true_period:
            score['correct'] += 1
        else:
            score['incorrect'] += 1

        detail[(true_period, guessed_period)] += 1

    eval_score = score['correct'] / (score['correct'] + score['incorrect'])
    print(composition, 'Score:', score, 'Evaluation score:', eval_score)
    
    return score, eval_score, detail

if __name__ == '__main__':
    