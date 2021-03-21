import python_minifier
import io
import tokenize
import re
import autopep8


_all__ = ('tokenize_r', )


def rm_and_tokenize(source: str) -> list:
    ''' removes comments and docstrings, tokenizes text, subs 4 spaces with tabs, returns tokeized list '''
    
    io_obj = io.StringIO(source)
    out = []
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out.append(" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out.append(token_string)
        else:
            out.append(token_string)
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = ['\t' if a == '    ' else a for a in out]
    return out


def get_file_contents(file: str) -> str:
    with open(file, 'r', encoding='utf-8') as f:
        code = f.read()
    return code


def tokenize_r(file=None, code=None) -> list or None:
    code = get_file_contents(file) if file else code
    if code:
        try:
            code = autopep8.fix_code(code,
                                options={'ignore': ['E402']})
            result = python_minifier.minify(
                source=code,
                remove_annotations=True,
                remove_pass=False,
                remove_literal_statements=True,
                combine_imports=False,
                hoist_literals=False,
                rename_locals=False,
                rename_globals=False,
                remove_object_base=False,
                convert_posargs_to_args=False,
            )
            out = rm_and_tokenize(code)
            return out
        except Exception as e:
            print(f"{e} {file}")
            with open('exceptions.log', 'a') as f:
                f.write(f"{e} {file}\n")
    return None

def tokenizer_py(file=None, code=None) -> list or None:
    code = get_file_contents(file) if file else code
    if code:
        try:
            # code = autopep8.fix_code(code,
            #                     options={'ignore': ['E402']})
#             result = python_minifier.minify(
#                 source=code,
#                 remove_annotations=True,
#                 remove_pass=False,
#                 remove_literal_statements=True,
#                 combine_imports=False,
#                 hoist_literals=False,
#                 rename_locals=False,
#                 rename_globals=False,
#                 remove_object_base=False,
#                 convert_posargs_to_args=False,
#             )
            out = rm_and_tokenize(code)
            return out
        except Exception as e:
            print(f"{e} {file}")
            with open('exceptions.log', 'a') as f:
                f.write(f"{e} {file}\n")
    return None

code = '''
def bmi(height: "Meters", weight: "Kgs"):
    bmi = weight / (height**2)
    print("Your BMI is: {0} and you are ".format(bmi), end='')
    if (bmi < 16):
       print("severely underweight.")
    elif (bmi >= 16 and bmi < 18.5):
       print("underweight.")
    elif (bmi >= 18.5 and bmi < 25):
       print("healthy.")
    elif (bmi >= 25 and bmi < 30):
       print("overweight.")
    elif (bmi >= 30):
       print("severely overweight.")
'''
tokenize_r(code=code)