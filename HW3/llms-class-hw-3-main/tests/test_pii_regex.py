from src.regex_cleaning import clean_pii


def test_phone_number_1():
    dirty_string = 'CMU\'s phone number is (412) 268-2000' 
    clean_string = 'CMU\'s phone number is '

    assert clean_pii(dirty_string) == clean_string

def test_phone_number_2():
    dirty_string = 'CMU\'s phone number is 412-268-2000' 
    clean_string = 'CMU\'s phone number is '

    assert clean_pii(dirty_string) == clean_string

def test_phone_number_3():
    dirty_string = '(412)268-2000' 
    clean_string = ''

    assert clean_pii(dirty_string) == clean_string

def test_phone_number_4():
    dirty_string = '+14122682000' 
    clean_string = ''

    assert clean_pii(dirty_string) == clean_string

def test_email_1():
    dirty_string = 'email me at abertsch@cs.cmu.edu'
    clean_string = 'email me at '
    assert clean_pii(dirty_string) == clean_string

def test_email_2():
    dirty_string = 'randomemail@gmail.com'
    clean_string = ''
    assert clean_pii(dirty_string) == clean_string
    
def test_non_pii_1():
    dirty_string = 'come see me @work tmr'
    assert clean_pii(dirty_string) == dirty_string

def test_non_pii_2():
    dirty_string = 'I have 205,124,126 tokens in one dataset'
    assert clean_pii(dirty_string) == dirty_string

def test_non_pii_3():
    dirty_string = 'there are between 142-315 cookies left after the bake sale'
    assert clean_pii(dirty_string) == dirty_string

def test_snn_1():
    dirty_string = '123-45-6789'
    clean_string = ''
    assert clean_pii(dirty_string) == clean_string
