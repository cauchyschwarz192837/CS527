import urllib.request
import ssl
from os import path as osp
import shutil
import numpy as np
import skimage

def make_image(sentence, shape, letter_templates, white_col):
    fin_img = np.empty(shape=shape)
    fin_img = np.expand_dims(fin_img, axis = 1)

    word_space = np.concatenate((white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col\
        , white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col\
        , white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col), axis = 1)

    for k, letter in enumerate(sentence):
        if letter == " ":
            curr = word_space
        else:
            curr = letter_templates[letter]
            start = 0
            end = 0
            for i in range(curr.shape[1]):
                if np.min(curr[:, i]) < 250:
                    start = i
                    break
            for i in range(curr.shape[1]):
                if np.min(curr[:, curr.shape[1] - 1 - i]) < 250:
                    end = curr.shape[1] - i
                    break
            curr = curr[:, start:end]

        fin_img = np.concatenate((fin_img, curr), axis = 1)

    return fin_img


def retrieve(file_name, semester='spring25', homework=1):
    if osp.exists(file_name):
        print('Using previously downloaded file {}'.format(file_name))
    else:
        context = ssl._create_unverified_context()
        fmt = 'https://www2.cs.duke.edu/courses/{}/compsci527/homework/{}/{}'
        url = fmt.format(semester, homework, file_name)
        with urllib.request.urlopen(url, context=context) as response:
            with open(file_name, 'wb') as file:
                shutil.copyfileobj(response, file)
        print('Downloaded file {}'.format(file_name))

def show_text_image(image):
    plt.figure(figsize=(9, 2), tight_layout=True, facecolor='lightgray')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.draw()
    plt.show()

if __name__ == "__main__":

    from os import path as osp
    from contextlib import redirect_stdout
    from os import makedirs

    text_image_file_name = 'text.png'
    alphabet_directory_name = 'alphabet'
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    retrieve(text_image_file_name)
    makedirs(alphabet_directory_name, exist_ok=True)
    print('Retrieving alphabet if needed')
    with redirect_stdout(None):
        for letter in alphabet:
            filename = alphabet_directory_name + '/' + '.'.join((letter, 'png'))
            retrieve(filename)

    from matplotlib import pyplot as plt
    # %matplotlib inline
    import skimage.io as io

    text_image = io.imread(text_image_file_name)
    show_text_image(text_image)

    letter_templates = {c: io.imread(osp.join(alphabet_directory_name, c + '.png')) for c in alphabet}

    from matplotlib.gridspec import GridSpec

    aspects = [i.shape[1] / i.shape[0] for i in letter_templates.values()]
    rows, cols = 1, 26
    gs = GridSpec(rows, cols, width_ratios=aspects)
    fig = plt.figure(figsize=(9, 2), tight_layout=True, facecolor='lightgray')
    for k, letter in enumerate(alphabet):
        plt.subplot(gs[k])
        plt.imshow(letter_templates[letter], cmap='gray')
        plt.axis('off')
    fig.subplots_adjust(left=0, right=1, wspace=0)
    plt.show()

    #-----------------------------------------------------------------------------------
    correct_sentence = 'the quick brown fox jumps over the lazy dog'
    #-----------------------------------------------------------------------------------

    indics = np.zeros(shape=text_image.shape[1])
    for i in range(text_image.shape[1]):
        if np.min(text_image[:, i]) < 250:
            indics[i] = 1

    letter_inds = []
    space_inds = []

    mov = 0
    start = 0
    mode = False

    while(mov < len(indics)):
        if (not mode): # on whitespaces
            if mov + 1 < len(indics) and indics[mov + 1] != indics[mov]:
                mode = not mode
                space_inds.append((start, mov))
                start = mov + 1
            mov += 1

        else: # on blackletters
            if mov + 1 < len(indics) and indics[mov + 1] != indics[mov]:
                mode = not mode
                letter_inds.append((start, mov))
                start = mov + 1
            mov += 1

    #-----------------------------------------------------------------------------------

    final_spaces = np.array([])
    for i in range(len(space_inds)):
        start = space_inds[i][0]
        end = space_inds[i][1]
        if (end - start + 1) > 15:
            final_spaces = np.append(final_spaces, i)

    #-----------------------------------------------------------------------------------

    word_inds = []

    i = 0
    while i < len(letter_inds):
        start, end = letter_inds[i]

        while (i + 1) < len(letter_inds) and ((letter_inds[i + 1][0] - 1) - (end + 1) + 1 <= 15):
            end = letter_inds[i + 1][1]
            i += 1

        word_inds.append((start, end))
        i += 1

    #-----------------------------------------------------------------------------------

    whichletter = []
    white_col = np.full(text_image.shape[0], 255)
    white_col = np.expand_dims(white_col, axis = 1)
    for elem in letter_inds:
        extract_im = text_image[:, elem[0] : elem[1] + 1]
        white_cols = np.concatenate((white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col\
            , white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col\
            , white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col, white_col), axis = 1)
        padded_extract_im = np.concatenate((white_cols, extract_im, white_cols), axis = 1)

        similarities = []
        for k, letter in enumerate(alphabet):
            maxsim = np.max(skimage.feature.match_template(padded_extract_im, letter_templates[letter], pad_input=False))
            similarities.append(maxsim)

        whichletter.append(np.argmax(similarities))

    #-----------------------------------------------------------------------------------

    final_spaces = np.delete(final_spaces, 0)
    guess_sentence = ""
    iter_i = 0
    while iter_i < len(whichletter):
        if iter_i in final_spaces:
            guess_sentence = guess_sentence + " "
            final_spaces = np.delete(final_spaces, 0)

        else:
            guess_sentence = guess_sentence + alphabet[whichletter[iter_i]]
            iter_i += 1

    #-----------------------------------------------------------------------------------

    # printed sentence
    print(guess_sentence)

    #-----------------------------------------------------------------------------------

    # show constructed image
    fin_img = make_image(guess_sentence, text_image.shape[0], letter_templates, white_col)
    show_text_image(fin_img)

    #-----------------------------------------------------------------------------------

    retrieve('edit_distance.py')