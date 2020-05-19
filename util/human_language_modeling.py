import torch
import logging
import io
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.data.functional import numericalize_tokens_from_iterator
import re
import spacy

URLS = {
    'WikiText2':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
    'WikiText103':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
    'PennTreebank':
        ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt'],
    'HumanNumbers':
        'http://files.fast.ai/data/examples/human_numbers'
}


NLP = spacy.load('en_core_web_sm')
def tokenizer(comment):
    #comment = re.sub(
    #   r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;\.]", " ", 
    #   str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    #comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    cleanr = re.compile('<.*?>')
    comment = re.sub(cleanr, '', comment)
    #if (len(comment) > MAX_CHARS):
    #   comment = comment[:MAX_CHARS]
    return[x.text for x in NLP.tokenizer(comment) if x.text != " "]


class LanguageModelingDataset(torch.utils.data.Dataset):
    """Defines a dataset for language modeling.
       Currently, we only support the following datasets:

             - WikiText2
             - WikiText103
             - PennTreebank

    """

    def __init__(self, data, vocab, raw_data):
        """Initiate language modeling dataset.

        Arguments:
            data: a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab: Vocabulary object used for dataset.

        Examples:
            >>> from torchtext.vocab import build_vocab_from_iterator
            >>> data = torch.tensor([token_id_1, token_id_2,
                                     token_id_3, token_id_1]).long()
            >>> vocab = build_vocab_from_iterator([['language', 'modeling']])
            >>> dataset = LanguageModelingDataset(data, vocab)

        """

        super(LanguageModelingDataset, self).__init__()
        self.data = data
        self.vocab = vocab
        self.raw_data = raw_data
        

    def __getitem__(self, i):
        return self.data[i]
    
    def __getrawdata__(self, i):
        return self.raw_data[i]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x

    def get_vocab(self):
        return self.vocab


class HumanLanguageModelingDataset(torch.utils.data.Dataset):
    """Defines a dataset for language modeling.
       Currently, we only support the following datasets:

             - WikiText2
             - WikiText103
             - PennTreebank
             

    """

    def __init__(self, input_data, vocab, label_data):
        """Initiate language modeling dataset.

        Arguments:
            data: a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab: Vocabulary object used for dataset.

        Examples:
            >>> from torchtext.vocab import build_vocab_from_iterator
            >>> data = torch.tensor([token_id_1, token_id_2,
                                     token_id_3, token_id_1]).long()
            >>> vocab = build_vocab_from_iterator([['language', 'modeling']])
            >>> dataset = LanguageModelingDataset(data, vocab)

        """

        super(HumanLanguageModelingDataset, self).__init__()
        self.input_data = input_data
        self.vocab = vocab
        self.label_data = label_data
        

    def __getitem__(self, i):
        return (self.input_data[i], self.label_data[i])

    def __len__(self):
        return len(self.input_data)

    def __iter__(self):
        for x in self.input_data:
            yield x

    def get_vocab(self):
        return self.vocab


def _get_datafile_path(key, extracted_files):
    for fname in extracted_files:
        if key in fname:
            return fname


def _setup_datasets(dataset_name, tokenizer=tokenizer,
                    root='.data', vocab=None, removed_tokens=[],
                    data_select=('train', 'test', 'valid'),bptt=None,batch_size=64):

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'test', 'valid'))):
        raise TypeError('data_select is not supported!')
    
    print (tokenizer)
    if tokenizer is None:
        tokenizer = get_tokenizer('basic_english')

    if dataset_name == 'PennTreebank':
        extracted_files = []
        select_to_index = {'train': 0, 'test': 1, 'valid': 2}
        extracted_files = [download_from_url(URLS['PennTreebank'][select_to_index[key]],
                                             root=root) for key in data_select]
    elif dataset_name == 'HumanNumbers':
        extracted_files = ['/Users/ssaurabh/.fastai/data/human_numbers/train.txt',
                         '/Users/ssaurabh/.fastai/data/human_numbers/valid.txt']
    else:
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
        extracted_files = extract_archive(dataset_tar)

    _path = {}
    for item in data_select:
        _path[item] = _get_datafile_path(item, extracted_files)
        
    #print(_path)

    if vocab is None:
        if 'train' not in _path.keys():
            raise TypeError("Must pass a vocab if train is not selected.")
        logging.info('Building Vocab based on {}'.format(_path['train']))
        txt_iter = iter(tokenizer(row) for row in io.open(_path['train'],
                                                          encoding="utf8"))
        vocab = build_vocab_from_iterator(txt_iter)
        logging.info('Vocab has {} entries'.format(len(vocab)))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")

    data = {}
    raw_data = {}
    for item in _path.keys():
        data[item] = []
        raw_data[item] = []
        logging.info('Creating {} data'.format(item))
        txt_iter = iter(tokenizer(row) for row in io.open(_path[item],
                                                          encoding="utf8"))
        
        for txt in txt_iter:
            raw_data[item] += txt
        
        txt_iter = iter(tokenizer(row) for row in io.open(_path[item],
                                                          encoding="utf8"))
        _iter = numericalize_tokens_from_iterator(
            vocab, txt_iter, removed_tokens)
        for tokens in _iter:
            data[item] += [token_id for token_id in tokens]

    for key in data_select:
        if data[key] == []:
            raise TypeError('Dataset {} is empty!'.format(key))
      
    if bptt is None:
        return tuple(LanguageModelingDataset(torch.tensor(data[d]).long(), vocab, raw_data[d])
                     for d in data_select)
    else:
        #### generate input and labels
        input_data = {}
        label_data = {}
        for key in data_select:
            #### Extend the dataset such that the last batch is not left out
            recycled_data_len = (bptt*batch_size) - (len(data[key])%(bptt*batch_size)) + bptt
            data[key] = data[key] + data[key][0:recycled_data_len]
            input_d = []
            label_d = []
            for i in range(len(data[key]) - bptt):
                input_d.append(data[key][i:i+bptt])
                label_d.append(data[key][i+1:i+bptt+1])
            print (len(input_d))
            print (len(label_d))
            input_data[key] = torch.tensor(input_d)
            label_data[key] = torch.tensor(label_d)
            print(input_data[key].shape)
            print(label_data[key].shape)
                                         
                                        
            #input_d = torch.tensor(data[key]).long()
            ###reshape the input data
            #if input_d.shape[0]%bptt > 0:
            #    pad_len_input = bptt - input_d.shape[0]%bptt
            #else:
            #    pad_len_input = 0
            #input_d = torch.nn.functional.pad(input_d, (0, pad_len_input), mode='constant', value=1)
            
            #input_data[key] = input_d.reshape(int(input_d.shape[0]/bptt),bptt)
            
            
            #if input_d[1:].shape[0]%bptt > 0:
            #    pad_len_output = bptt - input_d[1:].shape[0]%bptt
            #else:
            #    pad_len_output = 0
            
            #input_d = torch.nn.functional.pad(input_d[1:], (0, pad_len_output), mode='constant', value=1)
            #label_data[key] = input_d.reshape(int(input_d.shape[0]/bptt),bptt) 
            
        return tuple(HumanLanguageModelingDataset(input_data[d], vocab, label_data[d]) for d in data_select)
            
            



def HumanNumbers(*args, **kwargs):
    """ Defines WikiText2 datasets.

    Create language modeling dataset: WikiText2
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: [])
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.
        bptt: Memory for LSTM and GRU

    Examples:
        >>> from torchtext.experimental.datasets import HumanNumbers
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = WikiText2(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = WikiText2(tokenizer=tokenizer, vocab=vocab,
                                       data_select='valid')

    """

    return _setup_datasets(*(("HumanNumbers",) + args), **kwargs)



def WikiText2(*args, **kwargs):
    """ Defines WikiText2 datasets.

    Create language modeling dataset: WikiText2
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: [])
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import WikiText2
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = WikiText2(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = WikiText2(tokenizer=tokenizer, vocab=vocab,
                                       data_select='valid')

    """

    return _setup_datasets(*(("WikiText2",) + args), **kwargs)


def WikiText103(*args, **kwargs):
    """ Defines WikiText103 datasets.

    Create language modeling dataset: WikiText103
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        data_select: the returned datasets (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test').
            If 'train' is not in the tuple, an vocab object should be provided which will
            be used to process valid and/or test data.
        removed_tokens: removed tokens from output dataset (Default: [])
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import WikiText103
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = WikiText103(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = WikiText103(tokenizer=tokenizer, vocab=vocab,
                                         data_select='valid')

    """

    return _setup_datasets(*(("WikiText103",) + args), **kwargs)


def PennTreebank(*args, **kwargs):
    """ Defines PennTreebank datasets.

    Create language modeling dataset: PennTreebank
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: [])
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import PennTreebank
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = PennTreebank(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = PennTreebank(tokenizer=tokenizer, vocab=vocab,
                                          data_select='valid')

    """

    return _setup_datasets(*(("PennTreebank",) + args), **kwargs)
