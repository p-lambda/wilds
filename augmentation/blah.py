import random
import textwrap
from nltk.corpus import wordnet as wn
# import nltk
# nltk.download('wordnet')

def format_python_code(snippet):
    formatted_code = snippet.replace(" <EOL>", "\n")
    formatted_code = formatted_code.replace("<s>", "").replace("</s>", "")
    return formatted_code

def reformat_to_original_style(code):
    # Replace newlines with <EOL> and add <s> and </s>
    formatted_code = code.replace("\n", " <EOL>")
    return f"<s> {formatted_code} </s>"

def insert_safe_random_spaces(code, n):
    """
    Inserts `n` random spaces into safe locations in the given code snippet.
    Safe locations include spaces around operators, after commas, and around braces and brackets.

    :param code: String containing the Python code snippet.
    :param n: Number of random spaces to insert.
    :return: String with the code snippet having `n` random spaces inserted at safe locations.
    """
    if n <= 0:
        return code

    # Characters around which spaces can safely be inserted
    safe_chars = set(",;()[]{}+-*/&|=<>")

    # Identifying safe positions to insert spaces
    safe_positions = [i for i, char in enumerate(code) if char in safe_chars]

    for _ in range(n):
        if safe_positions:
            # Choosing a random position from the safe positions
            pos = random.choice(safe_positions)
            code = code[:pos] + ' ' + code[pos:]
            # Updating safe positions after insertion
            safe_positions = [i + 1 if i >= pos else i for i in safe_positions]
        else:
            # If no safe positions left, break the loop
            break

    return code

##############################################################################

def get_related_words(word, pos=wn.NOUN, limit=3):
    synsets = wn.synsets(word, pos=pos)
    related_words = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            if len(related_words) < limit:
                related_words.add(lemma.name().replace('_', ' '))
            else:
                break
    return list(related_words)

def generate_function_name():
    """Generate a function name using a random noun from WordNet."""
    synsets = list(wn.all_synsets(wn.NOUN))
    word = random.choice(synsets).lemmas()[0].name().replace('_', '')
    return word

def generate_random_string_value(base_word):
    """Generate a random string value related to a base word."""
    related_words = get_related_words(base_word, limit=10)
    if related_words:
        return random.choice(related_words)
    else:
        # Fallback to a generic string if no related words are found
        return "related string"

def generate_random_value(base_word):
    """Generate a random value (could be an integer, float, or a related string)."""
    value_types = [
        lambda: random.randint(0, 100),
        lambda: round(random.uniform(0.0, 100.0), 2),
        lambda: generate_random_string_value(base_word)
    ]
    return random.choice(value_types)()

def generate_related_var_name(arg_name):
    """Generate a variable name related to the argument name."""
    related_words = get_related_words(arg_name, limit=10)
    for word in related_words:
        if word.replace(' ', '_') != arg_name:
            return word.replace(' ', '_')
    return arg_name + '_var'  # Fallback if no different name is found

def get_synonyms(word):
    """Fetch synonyms of a given word."""
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

def generate_function_purpose_comment(func_name, related_args):
    """Generate a tailored purpose statement for the function."""
    action_words = get_synonyms("process")
    complexity_words = get_related_words(func_name, limit=5)
    complexity_word = random.choice(complexity_words) if complexity_words else "complex"
    action_word = random.choice(action_words) if action_words else "processes"

    args_description = ", ".join(related_args)
    return f"    # The function '{func_name}' {action_word} {args_description} for {complexity_word} tasks."

def generate_comment_for_var(var_name, func_name):
    """Generate a specific comment for a variable in the context of the function."""
    action_words = get_synonyms("operation")
    action_word = random.choice(action_words) if action_words else "operation"
    return f"    # {var_name} is utilized in {func_name} {action_word}."

def generate_random_function():
    func_name = generate_function_name()
    related_args = get_related_words(func_name, pos=wn.NOUN, limit=random.randint(1, 3))
    args = ', '.join([arg.replace(' ', '_') for arg in related_args])
    
    body_lines = [generate_function_purpose_comment(func_name, related_args)]
    operation_lines = []
    for arg in related_args:
        related_var_name = generate_related_var_name(arg.replace(' ', '_'))
        operation_value = repr(generate_random_value(func_name))
        operation_line = f"    {related_var_name} = {arg.replace(' ', '_')} + {operation_value}"
        if random.choice([True, False]):  # Randomly decide to add a comment
            comment = generate_comment_for_var(related_var_name, func_name)
            operation_line += " " + comment  # Append comment on the same line
        operation_lines.append(operation_line)
        body_lines.append(operation_line)

    if operation_lines:
        num_vars_to_return = random.randint(1, len(operation_lines))
        return_vars = random.sample(operation_lines, num_vars_to_return)
        return_statement = 'return ' + ', '.join([var.split('=')[0].strip() for var in return_vars])
        body_lines.append(return_statement)

    function_def = f"def {func_name}({args}):\n" + '\n'.join(body_lines)
    return function_def

def insert_random_functions(code, n):
    # Split the code into lines
    lines = code.split('\n')
    
    # Decide on insertion points
    # For simplicity, insert at the end of the code
    insertion_point = len(lines)
    
    # Generate and insert n functions
    for _ in range(n):
        random_func = generate_random_function()
        lines.insert(insertion_point, textwrap.dedent(random_func))
        insertion_point += len(random_func.split('\n'))
    
    # Combine back into a single string
    return '\n'.join(lines)


##############################################################################

snippet = "<s> import sys , operator , string , os , threading , re <EOL> from util import getch , cls , get_input <EOL> from time import sleep <EOL> lock = threading . Lock ( ) <EOL> class FreqObserver ( threading . Thread ) : <EOL> def __init__ ( self , freqs ) : <EOL> threading . Thread . __init__ ( self ) <EOL> self . daemon , self . _end = True , False <EOL> self . _freqs = freqs <EOL> self . _freqs_0 = sorted ( self . _freqs . iteritems ( ) , key = operator . itemgetter ( 1 ) , reverse = True ) [ : 25 ] <EOL> self . start ( ) <EOL> def run ( self ) : <EOL> while not self . _end : <EOL> self . _update_view ( ) <EOL> sleep ( 0.1 ) <EOL> self . _update_view ( ) <EOL> def stop ( self ) : <EOL> self . _end = True <EOL> def _update_view ( self ) : <EOL> lock . acquire ( ) <EOL> freqs_1 = sorted ( self . _freqs . iteritems ( ) , key = operator . itemgetter ( 1 ) , reverse = True ) [ : 25 ] <EOL> lock . release ( ) <EOL> if ( freqs_1 != self . _freqs_0 ) : <EOL> self . _update_display ( freqs_1 ) <EOL> self . _freqs_0 = freqs_1 <EOL> def _update_display ( self , tuples ) : <EOL> def refresh_screen ( data ) : <EOL> cls ( ) <EOL> print data <EOL> sys . stdout . flush ( ) <EOL> data_str = "" <EOL> for ( w , c ) in tuples : <EOL> data_str += str ( w ) + ' - ' + str ( c ) + '\n' <EOL> refresh_screen ( data_str ) <EOL> class WordsCounter : <EOL> freqs = { } <EOL> def count ( self ) : <EOL> def non_stop_words ( ) : <EOL> stopwords = set ( open ( '' ) . read ( ) . split ( ',' ) + list ( string . ascii_lowercase ) ) <EOL> for line in f : <EOL> yield [ w for w in re . findall ( '[a-z]{2,}' , line . lower ( ) ) if w not in stopwords ] <EOL> words = non_stop_words ( ) . next ( ) <EOL> lock . acquire ( ) <EOL> for w in words : <EOL> self . freqs [ w ] = 1 if w not in self . freqs else self . freqs [ w ] + 1 <EOL> lock . release ( ) <EOL> print "" <EOL> print "" <EOL> model = WordsCounter ( ) <EOL> view = FreqObserver ( model . freqs ) <EOL> with open ( sys . argv [ 1 ] ) as f : <EOL> while get_input ( ) : <EOL> try : <EOL> model . count ( ) <EOL> except StopIteration : <EOL> view . stop ( ) <EOL> sleep ( 1 ) <EOL> break </s>"

# Format and indent
formated_snippet = format_python_code(snippet)
# print("Formated Snippet:\n", formated_snippet)

augmented_code = insert_random_functions(formated_snippet, 2)
# print("\nAugmented Snippet:\n", augmented_code)

print("Origianl snippet:\n", snippet)
# Revert to original format
reverted_snippet = reformat_to_original_style(augmented_code)
print("\nReverted Snippet:\n", reverted_snippet)