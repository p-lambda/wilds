import os, random
from shutil import copyfile
from refactoring_methods import *
import nltk


def return_function_code(code, method_names):
    """
    Extract functions from the provided code that match given method names.

    Args:
        code (str): The source code containing classes and methods.
        method_names (list of str): Method names to search for within the code.

    Returns:
        tuple: Two lists, one with the code of the matching functions and 
               another with their corresponding names.
    """
    final_codes = []
    final_names = []
    Class_list, raw_code = extract_class(code)
    for class_name in Class_list:
        function_list, class_name = extract_function(class_name)
    for fun_code in function_list:
        for method_name in method_names:
            method_name_tem = method_name.replace('|', '')
            if method_name_tem.upper() in fun_code.split('\n')[0].upper():

                final_codes.append(fun_code)
                final_names.append(method_name)
    return final_codes, final_names


def generate_adversarial(k, code, method_names):
    """
    Generate refactored code by applying k different refactoring methods to functions
    matching the specified method names within the provided code.

    This function focuses on refactoring at the method level, targeting specific methods
    within classes extracted from the code. It is intended for precise, targeted refactoring
    when you want to alter specific methods identified by name.

    Args:
        k (int): The number of refactoring methods to apply to each targeted method.
        code (str): The source code containing classes and methods to refactor.
        method_names (list of str): Specific method names within the code to target for refactoring.

    Returns:
        str: The refactored code with transformations applied to the specified methods.
    """
    method_name = method_names[0]
    function_list = []
    class_name = ''
    Class_list, raw_code = extract_class(code)
    for class_name in Class_list:
        function_list, class_name = extract_function(class_name)

    refac = []
    new_refactored_code = ''
    for code in function_list:
        if method_name not in code.split('\n')[0]:
            continue
        new_rf = code
        new_refactored_code = code
        for t in range(k):
            refactors_list = [rename_argument,
                                return_optimal,
                                add_argumemts,
                                rename_api,
                                rename_local_variable,
                                add_local_variable,
                                rename_method_name,
                                enhance_if,
                                add_print,
                                duplication,
                                apply_plus_zero_math,
                                dead_branch_if_else,
                                dead_branch_if,
                                dead_branch_while,
                                dead_branch_for,
                                # dead_branch_switch
                                ]#
            vv = 0
            while new_rf == new_refactored_code and vv <= 20:
                try:
                    vv += 1
                    refactor       = random.choice(refactors_list)
                    print('*'*50 , refactor , '*'*50)
                    new_refactored_code = refactor(new_refactored_code)

                except Exception as error:
                    print('error:\t', error)

            new_rf = new_refactored_code
            print('----------------------------OUT of WHILE----------------------------------', vv)
            print('----------------------------CHANGED THJIS TIME:----------------------------------', vv)
        refac.append(new_refactored_code)
    code_body = raw_code.strip() + ' ' + class_name.strip()
    for i in range(len(refac)):
        final_refactor = code_body.replace('vesal' + str(i), str(refac[i]))
        code_body = final_refactor
    return new_refactored_code


def generate_adversarial_json(k, code):
    """
    Generate a list of refactored code snippets by applying k different refactoring methods.

    This function is a variant of the adversarial generation that is intended to produce
    a collection of refactored code snippets, potentially for further processing or analysis.
    The current implementation does not target specific methods or produce JSON output, but
    the name suggests it may be extended for such purposes in the future.

    Args:
        k (int): The number of refactoring methods to apply to the code snippets.
        code (str): The source code to refactor.

    Returns:
        list of str: A list of refactored code snippets, each potentially
                     altered by the refactoring methods.
    """
    final_refactor = ''
    function_list = []
    class_name = ''
    vv = 0
    if len(function_list) == 0:
        function_list.append(code)
    refac = []
    for code in function_list:
        new_rf = code
        new_refactored_code = code
        for t in range(k):
            refactors_list = [rename_argument,
                              return_optimal,
                              add_argumemts,
                              rename_api,
                              rename_local_variable,
                              add_local_variable,
                              rename_method_name,
                              enhance_if,
                              add_print,
                              duplication,
                              apply_plus_zero_math,
                              dead_branch_if_else,
                              dead_branch_if,
                              dead_branch_while,
                              dead_branch_for,
                              # dead_branch_switch
                              ]  
            vv = 0
            while new_rf == new_refactored_code and vv <= 20:
                try:
                    vv += 1
                    refactor = random.choice(refactors_list)
                    print('*' * 50, refactor, '*' * 50)
                    new_refactored_code = refactor(new_refactored_code)

                except Exception as error:
                    print('error:\t', error)

            new_rf = new_refactored_code
        refac.append(new_refactored_code)

    print("refactoring finished")
    return refac


def generate_adversarial_file_level(k, code):
    """
    Apply k refactoring operations to the entire file-level code, potentially altering the overall structure.

    This function applies refactoring transformations to the code at the file level as a whole,
    without targeting specific methods or classes. It is suitable for broad refactoring that impacts
    the entire codebase, such as renaming variables used across multiple methods or classes, or
    adding additional error handling throughout the file.

    Args:
        k (int): The number of refactoring methods to apply to the entire code.
        code (str): The source code of the entire file to refactor.

    Returns:
        str: The refactored code at the file level, with each transformation potentially affecting the global scope.
    """
    new_refactored_code = ''
    new_rf = code
    new_refactored_code = code
    for t in range(k):
        refactors_list = [
                            rename_argument, 
                            return_optimal, 
                            add_argumemts,
                            rename_api, 
                            rename_local_variable,
                            add_local_variable,
                            rename_method_name,
                            enhance_if,
                            add_print,
                            duplication,
                            apply_plus_zero_math,
                            dead_branch_if_else,
                            dead_branch_if,
                            dead_branch_while,
                            dead_branch_for
                            ]  
        vv = 0
        while new_rf == new_refactored_code and vv <= 20:
            try:
                vv += 1
                refactor = random.choice(refactors_list)
                print('*' * 50, refactor, '*' * 50)
                new_refactored_code = refactor(new_refactored_code)
            except Exception as error:
                print('error:\t', error)
        new_rf = new_refactored_code
    return new_refactored_code

def generate_adversarial_file_level_n(code, n):
    refactors_list = [
                        rename_argument, 
                        return_optimal, 
                        add_argumemts,
                        rename_api, 
                        rename_local_variable,
                        add_local_variable,
                        rename_method_name,
                        enhance_if,
                        add_print,
                        duplication,
                        apply_plus_zero_math,
                        dead_branch_if_else,
                        dead_branch_if,
                        dead_branch_while,
                        dead_branch_for,
                        ]

    new_refactored_code = code
    refactoring_counts = {refactor.__name__: 0 for refactor in refactors_list}

    for refactor in refactors_list:
        attempts = 0
        while attempts < n:
            try:
                new_refactored_code = refactor(new_refactored_code)
                refactoring_counts[refactor.__name__] += 1
                attempts += 1
            except Exception as error:
                # Try all alternative methods before moving to the next refactoring
                for alternative_refactor in [rf for rf in refactors_list if rf != refactor]:
                    try:
                        new_refactored_code = alternative_refactor(new_refactored_code)
                        refactoring_counts[alternative_refactor.__name__] += 1
                        # print(f'Applied alternative {alternative_refactor.__name__}')
                        break  # Break if successful with an alternative
                    except Exception as alt_error:
                        # print(f'Error applying alternative {alternative_refactor.__name__}: {alt_error}')
                        continue  # Continue trying other alternatives if this one fails
                
                if refactoring_counts[alternative_refactor.__name__] == 0:
                    print(f'All alternatives failed for {refactor.__name__}')
                break  # Move to the next refactoring method after trying all alternatives

    return new_refactored_code, refactoring_counts

if __name__ == '__main__':
    """
    Main execution point of the script.

    Reads a Python file, applies refactoring, and prints the new code.
    """
    K = 1
    filename = 'blah.py'
    open_file = open(filename, 'r', encoding='ISO-8859-1')
    code = open_file.read()
    new_code = generate_adversarial_file_level(K, code)
    print(new_code)
