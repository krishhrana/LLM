from src.regex_cleaning import filter_noneng


def test_japanese():
    original_string = 'これは英語ではありません' 

    assert filter_noneng(original_string)

def test_chinese():
    original_string = '这不是英文' 

    assert filter_noneng(original_string)


def test_vietnamese():
    original_string = 'đây là tiếng việt' 

    assert filter_noneng(original_string)


def test_russian():
    original_string = 'это не английский' 

    assert filter_noneng(original_string)

def test_spanish():
    original_string = '¿Esto será eliminado?' 

    assert filter_noneng(original_string)


def test_spanish_2():
    original_string = 'este idioma es divertido' 

    assert not filter_noneng(original_string)


def test_english_1():
    original_string = 'hi my name is Samuel and this sentence is in English' 

    assert not filter_noneng(original_string)


def test_mixture_1():
    original_string = 'Pyotr Ilyich Tchaikovsky (Петр Ильич Чайковский) is known for his ballet music.' 

    assert filter_noneng(original_string)


def test_mixture_2():
    original_string = 'đây là tiếng việt and this is in the same sentence.' 

    assert filter_noneng(original_string)


def test_ascii_art():
    original_string = '°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸' 

    assert filter_noneng(original_string)

