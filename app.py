from analyzer import Analyzer
import traceback
from logger import Logger

lg = Logger()

try:
    anal = Analyzer()
    lg.log('Stage 1 - Started...')
    anal.stage_1("result.csv")
    lg.log('Stage 1 - Successfully completed')
except Exception as e:
    error = str(traceback.print_exc())
    lg.log(f'Stage 1 - Error: {e},\n Error logs: {error}')
