import random
from enum import Enum
from pathlib import Path
from sympy.parsing.sympy_parser import parse_expr

class RuleType(Enum):
    NONREPEATABLE = 1 # non repeatable
    REPEATABLE = 2 # repeatable with different values

class Rule():
    def __init__(self, args):
        pass

    def generate_rule(self, input):
        pass

    def validate(self, input):
        pass

    def generate_prompt(self):
        pass


# Rule 1
class CountNumCharRule(Rule):
    def __init__(self, args):
        self.num_char = None

    def generate_rule(self, input):
        self.num_char = len(input)
        return input

    def validate(self, input):
        return len(input.strip()) == self.num_char
    
    def generate_prompt(self):
        return f"the text has only {self.num_char} characters"
    

# Rule 2
class CountNumUppercaseCharRule(Rule):
    def __init__(self, args):
        self.num_char = None

    def generate_rule(self, input):
        count = 0
        for c in input:
            if "A" <= c <= "Z":
                count += 1
        
        self.num_char = count

        return input
    
    def validate(self, input):
        count = 0
        for c in input:
            if "A" <= c <= "Z":
                count += 1
        return count == self.num_char
    
    def generate_prompt(self):
        return f"the text has {self.num_char} uppercase characters"


# Rule 3
class CountNumLowercaseCharRule(Rule):
    def __init__(self, args):
        self.num_char = None
    
    def generate_rule(self, input):
        counts = 0
        for c in input:
            if "a" <= c <= "z":
                counts += 1

        self.num_char = counts
        return input

    def validate(self, input):
        count = 0
        for c in input:
            if "a" <= c <= "z":
                count += 1
        return count == self.num_char

    def generate_prompt(self):
        return f"the text has {self.num_char} lowercase characters"


# Rule 4
class CountNumSpecificCharRule(Rule):
    def __init__(self, args):
        self.num_char = None

        char_ord = random.randint(0, 25)
        is_upper = random.randint(0, 1)
        if is_upper:
            self.char = chr(ord("A") + char_ord)
        else:
            self.char = chr(ord("a") + char_ord)
    
    def generate_rule(self, input):
        counts = 0
        for c in input:
            if c == self.char:
                counts += 1
        self.num_char = counts

        return input
        
    def validate(self, input):
        counts = 0
        for c in input:
            if c == self.char:
                counts += 1
        return counts == self.num_char
    
    def generate_prompt(self):
        return f"the text has {self.num_char} '{self.char}' character"


# Rule 5
class CountNumEnglishAlphaRule(Rule):
    def __init__(self, args):
        self.extra_num_char = random.randint(args["min_extra_num_char"], args["max_extra_num_char"])
        self.num_char = None
    
    def generate_rule(self, input):
        counts = 0
        for c in input:
            if "a" <= c <= "z" or "A" <= c <= "Z":
                counts += 1
        self.num_char = counts + self.extra_num_char

        output = input
        for _ in range(self.extra_num_char):
            char_ord = random.randint(0, 25)
            is_upper = random.randint(0, 1)
            while is_upper and char_ord in [ord("I")-ord("A"), ord("V")-ord("A"), ord("X")-ord("A"), ord("L")-ord("A"), ord("C")-ord("A"), ord("D")-ord("A")]:
                char_ord = random.randint(0, 25)
                is_upper = random.randint(0, 1)
            if is_upper:
                output += chr(ord("A") + char_ord)
            else:
                output += chr(ord("a") + char_ord)
        return output

    def validate(self, input):
        counts = 0
        for c in input:
            if "a" <= c <= "z" or "A" <= c <= "Z":
                counts += 1
        return counts == self.num_char
    
    def generate_prompt(self):
        return f"the text has {self.num_char} english character"


# Rule 6
class CountNumDigitRule(Rule):
    def __init__(self, args):
        self.extra_num_char = random.randint(args["min_extra_num_char"], args["max_extra_num_char"])
        self.num_char = None

    def generate_rule(self, input):
        counts = 0
        for c in input:
            if "0" <= c <= "9":
                counts += 1
        self.num_char = counts + self.extra_num_char

        output = input
        for _ in range(self.extra_num_char):
            output += str(random.randint(0, 9))

        return output

    
    def validate(self, input):
        counts = 0
        for c in input:
            if "0" <= c <= "9":
                counts += 1
        return counts == self.num_char
    
    def generate_prompt(self):
        return f"the text has {self.num_char} number digits"


# Rule 7
class CountNumSpecialCharRule(Rule):
    def __init__(self, args):
        self.extra_num_char = random.randint(args["min_extra_num_char"], args["max_extra_num_char"])
        self.num_char = None
        self.special_char_list = ["!", "@", "#", "$", "%", "^", "&", "*"]

    def generate_rule(self, input):
        counts = 0
        for c in input:
            if c in self.special_char_list:
                counts += 1
        self.num_char = counts + self.extra_num_char

        output = input
        for _ in range(self.extra_num_char):
            output += str(self.special_char_list[random.randint(0, 7)])

        return output
    
    def validate(self, input):
        counts = 0
        for c in input:
            if c in self.special_char_list:
                counts += 1
        return counts == self.num_char
    
    def generate_prompt(self):
        return f"the text has {self.num_char} special characters, including '!', '@', '#', '$', '%', '^', '&', '*'"


# Rule 8
class CountNumRomansDigitRule(Rule):
    def __init__(self, args):
        self.extra_num_char = random.randint(args["min_extra_num_char"], args["max_extra_num_char"])
        self.num_char = None
        self.ROMANS_CHARS = ["I", "V", "X", "L", "C", "D"]

    def generate_rule(self, input):
        counts = 0
        for c in input:
            if c in self.ROMANS_CHARS:
                counts += 1
        self.num_char = counts + self.extra_num_char

        output = input
        for _ in range(self.extra_num_char):
            output += str(self.ROMANS_CHARS[random.randint(0, 5)])

        return output
    
    def validate(self, input):
        counts = {}
        for roman_char in self.ROMANS_CHARS:
            counts[roman_char] = 0
        for c in input:
            if c in counts:
                counts[c] += 1
        
        num_roman_char = 0
        for roman_char in self.ROMANS_CHARS:
            num_roman_char += counts[roman_char]
        return num_roman_char == self.num_char
    
    def generate_prompt(self):
        return f"the text has {self.num_char} roman digits, including 'I', 'V', 'X', 'L', 'C', 'D'"


# Rule 9
class ConsistStrRule(Rule):
    def __init__(self, args):
        self.words = args["words"]
        self.str = self.words[random.randint(0, len(self.words)-1)]

    def generate_rule(self, input):
        output = input + self.str
        return output
    
    def validate(self, input):
        return self.str in input
    
    def generate_prompt(self):
        return f"the text has \"{self.str}\" string"
    

# Rule 10
class ConsistCapitalOfRule(Rule):
    def __init__(self, args):
        self.words = args["words"]
        self.str = self.words[random.randint(0, len(self.words)-1)]
        self.country_to_capital_map = args["country_to_capital_map"]

    def generate_rule(self, input):
        output = input + self.country_to_capital_map[self.str.lower()]
        return output
    
    def validate(self, input):
        return self.country_to_capital_map[self.str.lower()].lower() in input.lower()

    def generate_prompt(self):
        return f"the text has the capital city of {self.str}"
    

# Rule 11
class ConsistContinentOfRule(Rule):
    def __init__(self, args):
        self.words = args["words"]
        self.str = self.words[random.randint(0, len(self.words)-1)]
        self.country_to_continent_map = args["country_to_continent_map"]

    def generate_rule(self, input):
        output = input + self.country_to_continent_map[self.str.lower()]
        return output
    
    def validate(self, input):
        return self.country_to_continent_map[self.str.lower()].lower() in input.lower()
    
    def generate_prompt(self):
        return f"the text has the continent of {self.str}"
    

# Rule 12
class ConsistSynonymOfRule(Rule):
    def __init__(self, args):
        self.words = args["words"]
        self.str = self.words[random.randint(0, len(self.words)-1)]
        self.country_to_continent_map = args["word_to_synonym_map"]
    
    def generate_rule(self, input):
        output = input + self.str
        return output
    
    def validate(self, input):
        return self.country_to_continent_map[self.str].lower() in input.lower()
    
    def generate_prompt(self):
        return f"the text has the synonym of {self.str}"


# Rule 13
class ConsistAntonymOfRule(Rule):
    def __init__(self, args):
        self.words = args["words"]
        self.str = self.words[random.randint(0, len(self.words)-1)]
        self.country_to_continent_map = args["word_to_antonym_map"]
    
    def generate_rule(self, input):
        output = input + self.str
        return output

    def validate(self, input):
        return self.country_to_continent_map[self.str].lower() in input.lower()
    
    def generate_prompt(self):
        return f"the text has the antonym of {self.str}"

# Rule 14
class ArithmeticMathExpressionRule(Rule):
    def __init__(self, args):
        self.operators = ["+", '-', "/", "*"]

        self.max_num_operator = args["max_num_operator"]
        num_operator = random.randint(1, self.max_num_operator)
        expression = ""

        value = None

        while value is None or int(value) != value:
            restart = False
            for i in range(num_operator):
                if i == 0:
                    expression = str(random.randint(0, 9))
                next_num = random.randint(0, 9)
                next_operator = self.operators[random.randint(0,len(self.operators)-1)]
                expression += " " + next_operator + " " + str(next_num)

                if next_operator == "/" and next_num == 0:
                    restart = True

            if restart:
                continue

            value = parse_expr(expression)

        self.expression = expression
        self.num = value

    def generate_rule(self, input):
        output = input + str(self.num)
        return output

    def validate(self, input):
        return str(self.num) in input
    
    def generate_prompt(self):
        return f"the text has a number that equals to {self.expression}"
    

# Rule 15
class ArithmeticMathWordExpressionRule(Rule):
    def __init__(self, args):
        self.num_to_word = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine",
        }
        self.operators_words = ["plus", 'minus', "divided by", "multiply by"]
        self.operators = ["+", '-', "/", "*"]

        self.words_to_expressions_dict = {
            **{v: k for k, v in self.num_to_word.items()},
            **{w: o for w, o in zip(self.operators_words, self.operators)},
        }

        self.max_num_operator = args["max_num_operator"]
        num_operator = random.randint(1, self.max_num_operator)

        expression = ""
        word_expression = ""
        value = None

        while value is None or int(value) != value:
            restart = False
            for i in range(num_operator):
                if i == 0:
                    num = random.randint(0, 9)
                    word_expression = self.num_to_word[num]
                    expression = str(num)
                    
                next_num = random.randint(0, 9)
                next_num_word = self.num_to_word[next_num]

                next_operator_id = random.randint(0,len(self.operators_words)-1)
                next_operator_word = self.operators_words[next_operator_id]
                next_operator = self.operators[next_operator_id]

                expression += " " + next_operator + " " + str(next_num)
                word_expression += " " + next_operator_word + " " + next_num_word

                if next_operator == "/" and next_num == 0:
                    restart = True

            if restart:
                continue
            
            value = parse_expr(expression)
        self.expression = word_expression
        self.num = value

    def generate_rule(self, input):
        output = input + str(self.num)
        return output

    def validate(self, input):
        return str(self.num) in input
    
    def generate_prompt(self):
        return f"the text has a number that equals to {self.expression}"