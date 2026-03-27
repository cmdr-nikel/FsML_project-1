import pandas as pd
import re

mb_prefixes = set("ABNC")

core_re = re.compile(
    r'^(?P<prefix>[A-Z])?'      # 1st letter
    r'(?P<core>\d{10})'         # 10 digit: 3+3+2+2
    r'(?P<suffix>[A-Z0-9]*)$'   # any suffix
)