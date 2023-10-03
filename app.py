from analyzer import Analyzer
import traceback
from logger import Logger

import random

import json

lg = Logger()

anal = Analyzer()
# try:
#     lg.log('Stage 1 - Started...')
#     # anal.stage_1("result.csv")
#     anal.stage_1_save_db("stage_1.csv")
#     lg.log('Stage 1 - Successfully completed')
# except Exception as e:
#     error = str(traceback.print_exc())
#     lg.log(f'Stage 1 - Error: {e},\n Error logs: {error}')

try:
    lg.log('Stage 2 - Started...')
    anal.stage_2('stage_1.csv', 'stage_2_day.csv', 'day')
    anal.stage_2('stage_1.csv', 'stage_2_week.csv', 'week')
    anal.stage_2('stage_1.csv', 'stage_2_month.csv', 'month')
    anal.stage_2_save_db('stage_2_day.csv', 'category_day')
    anal.stage_2_save_db('stage_2_week.csv', 'category_week')
    anal.stage_2_save_db('stage_2_month.csv', 'category_month')
    lg.log('Stage 2 - Successfully completed')
except Exception as e:
    error = str(traceback.print_exc())
    lg.log(f'Stage 2 - Error: {e},\n Error logs: {error}')
# with open('./text.txt', 'r', encoding='utf-8') as file:
#     content = '\n'.join(file.readlines())
#     t = json.loads(content)
#     print(t)