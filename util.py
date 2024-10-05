from docx import Document

def meta(**kwargs):
    def decorate(function):
        setattr(function, "meta", kwargs)
        return function

    return decorate

