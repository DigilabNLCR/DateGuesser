# DateGuesser
DateGuesser is an extension of the [AuthorGuesser project](https://github.com/DigilabNLCR/AuthorGuesser). While AuthorGuesser is designed to "guess" the authorship of a given text, DateGuesser aims to "guess" dating of texts, posing different challanges. In this case, it is essential not to confuse the date-specifics with author-specifics. The project is build with ML models from package [scikit-lear](https://scikit-learn.org/stable/), using primarily data from the digital collections of the [National Library of the Czech Republic](https://www.en.nkp.cz/).

# Current Status

The project is now under developement.

Experimetns have been done on Heyduk

Next authors to be explored are Světlá, and Jirásek

## Problems
From the beginning, it was clear that date attribution is more difficult task than authorship. While authors argauably have their own style, periods as such do not. Sure, there are some tendencies in any given period—but facts such as new authors write at the same time when the old ones are still active (but do not always change their writing to the new tendencies)—make this task possibly ill-aimed. Still, it is worth exploring.

### First experimental result: Heyduk

Because of the challanges, the current team working on the project (not the one designing the initial research question) was sceptical in regards to possible success of such an endeavour. Nonetheless, some initial experiments have been run, using works of Czech poet Adolf Heyduk. This data has been preprocessed by The Institute for Czech Literature of the CAS and are available [here](https://data.ucl.cas.cz/s/To3TMSK2G6SC6KQ). Further, the data have been preprocessed by us to fit the needs of the project (namely, structured into delexicalised XML). In the first experiments, the same delexicalisation methods as with the AuthorGuesser were used, namely changing all autosemantic words to part-of-speech tags only.

The data distribution is a first problem. On the following chart ([interactive](https://public.flourish.studio/visualisation/20673921/)), you can see that for some periods, the data are very sparse.

![Heyduk data](https://github.com/DigilabNLCR/DateGuesser/blob/main/img/heyduk_data.png)

For that reason, we have used 10-year periods instead of 1-year periods. We have run 73 leave-one-out experiments (in each case using 72 compositions for training and 1 for evaluation). The chosen ML model was LinearSVC (sklearn.svm.LinearSVC). The following heatmap¨([interactive](https://public.flourish.studio/visualisation/20676148/)) shows cumulative results where chunks of 10 stanzas are used as input documents (in sum, there are 6572 stanzas in the dataset). Shorter compositions (the shortest has only 2 stnzas) were used as whole. The results are less then satisfactory.

![Heyduk results](https://github.com/DigilabNLCR/DateGuesser/blob/main/img/heyduk_results.png)

### Future lines of inquiry

The problem may be that exploring a developement of one author only cannot reveal any real date changes. It is possible that if the training dataset is densened with a pleyad of different authors, some literary period tendencies may surface. In addition, other approaches, delexicalisation methods, and feature extractions may be useful in reaching better results. In the end, however, the current team remains sceptical towards this line of inquiry.