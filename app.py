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

# try:
#     lg.log('Stage 2 - Started...')
#     anal.stage_2('stage_1.csv', 'stage_2_day.csv', 'day')
#     anal.stage_2('stage_1.csv', 'stage_2_week.csv', 'week')
#     anal.stage_2('stage_1.csv', 'stage_2_month.csv', 'month')
#     anal.stage_2_save_db('stage_2_day.csv', 'category_day')
#     anal.stage_2_save_db('stage_2_week.csv', 'category_week')
#     anal.stage_2_save_db('stage_2_month.csv', 'category_month')
#     lg.log('Stage 2 - Successfully completed')
# except Exception as e:
#     error = str(traceback.print_exc())
#     lg.log(f'Stage 2 - Error: {e},\n Error logs: {error}')

# try:
#     lg.log('Stage 3 - Started...')
#     anal.stage_3('stage_2_day.csv', 'stage_1.csv', 'stage_3_day.csv')
#     anal.stage_3('stage_2_week.csv', 'stage_1.csv', 'stage_3_week.csv')
#     anal.stage_3('stage_2_month.csv', 'stage_1.csv', 'stage_3_month.csv')
#     anal.stage_3_save_db('stage_3_day.csv', "extra_research_day")
#     anal.stage_3_save_db('stage_3_week.csv', "extra_research_week")
#     anal.stage_3_save_db('stage_3_month.csv', "extra_research_month")
#     lg.log('Stage 3 - Successfully completed')
# except Exception as e:
#     error = str(traceback.print_exc())
#     lg.log(f'Stage 3 - Error: {e},\n Error logs: {error}')

try:
    lg.log('Stage 4 - Started...')
    # anal.stage_4('stage_3_day.csv', 'stage_4_day.csv')
    # anal.stage_4_save_db('stage_4_day.csv', "deep_research_day")
    # anal.stage_4('stage_3_week.csv', 'stage_4_week.csv')
    anal.stage_4_save_db('stage_4_week.csv', "deep_research_week")
    anal.stage_4('stage_3_month.csv', 'stage_4_month.csv')
    anal.stage_4_save_db('stage_4_month.csv', "deep_research_month")
    lg.log('Stage 3 - Successfully completed')
except Exception as e:
    error = str(traceback.print_exc())
    lg.log(f'Stage 3 - Error: {e},\n Error logs: {error}')


# # with open('./text.txt', 'r', encoding='utf-8') as file:
# #     content = '\n'.join(file.readlines())
# #     t = json.loads(content)
# #     print(t)
