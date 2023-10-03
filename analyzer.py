import os
import csv
import multiprocessing
from multiprocessing.pool import ApplyResult
from pymongo import MongoClient
from pymongo.database import Database
from dotenv import find_dotenv, load_dotenv
from stages import summarize_article, categorize
from logger import Logger
from time import time, sleep
from openai.error import InvalidRequestError, RateLimitError, AuthenticationError
import random
from helpers import remove_non_numbers_regex
import json
import traceback

# find and load .env file
load_dotenv(find_dotenv())

class Analyzer:
    """Article analyzer using Langchain and GPT"""
    session: MongoClient
    db: Database
    collections: dict
    categories: list[str]
    apikeys: list[str]
    logger: Logger

    def __init__(self) -> None:
        self.logger = Logger()
        self.session = MongoClient(os.environ["MONGODB_URL"])
        self.db = self.session["news-test"]
        self.collections = {
            0: "All categories",
            1: "lawandcrimes",
            2: "web3",
            3: "entertainments",
            4: "sports",
            5: "artandfashions",
            6: "bizandfinances",
            7: "politics",
            8: "scienceandteches",
            9: "lifestyleandhealths",
            10: "gamings"
        }
        self.categories = [
            "Politics",
            "Business and Finance",
            "Entertainment",
            "Science and Technology",
            "Sports",
            "Crypto/Web3",
            "Gaming",
            "Law and Crime",
            "Lifestyle and Health",
            "Art and Fashion",
        ]
        with open('keys/keys.txt', 'r') as keys_file:
            # Maximum 50 processes
            self.apikeys = [line.strip() for line in keys_file.readlines()[:50]]
    
    def log_invalid_key(self, apikey):
        # remove invallid apikey from valid list
        if apikey in self.apikeys:
            self.apikeys.remove(apikey)
            with open('keys/keys.txt', 'w') as keys_file:
                for key in self.apikeys:
                    keys_file.write(key + '\n')

            # add apikey to invalid key file
            with open('keys/invalid_keys.txt', 'a') as invalid_file:
                invalid_file.write(apikey + '\n')

    def stage_1(self, csv_filename):
        os.remove(csv_filename) if os.path.exists(csv_filename) else None

        start_t = time()
        # to test
        collection_name = [self.collections[i] for i in range(1, 11)]
        articles = []
        article_count = 0

        for category in collection_name:
            collection = self.db[category]
            documents = collection.find()
            for document in documents:
                article_count += 1
                print(f"{article_count} : {category}: {document['siteName']}, {document['link']}")
                articles.append([document['article'], document['siteName'], document['link']])
        end_t = time()
        
        self.logger.log(f'Stage 1 - {len(articles)} articles uploaded in {end_t - start_t} seconds, start processing...')
        
        start_t = time()
        article_count = 0

        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'article',
                'answer',
                'Title',
                'Category',
                'Summary',
                'Importance 1 day',
                'Reasoning for 1 day score',
                'Importance 1 week',
                'Reasoning for 1 week score',
                'Importance 1 month',
                'Reasoning for 1 month score',
                'site_name',
                'link'
            ])
        sumarized_count = 0
        total = len(articles)
        while len(articles) != 0:
            pool = multiprocessing.Pool(processes=min(len(self.apikeys), 50))
            results: list[ApplyResult] = []

            for i, (article, site_name, link) in enumerate(articles):
                api_key = self.apikeys[i % len(self.apikeys)]  # Use a different API key for each process
                if len(self.apikeys) > len(articles):
                    api_key = random.choice(self.apikeys)
                results.append(pool.apply_async(stage_1_thread_handler, (api_key, article, site_name, link,)))
            articles = []
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                for result in results:
                    article_data = result.get()
                    if article_data[1] == 'APIKey_Error':
                        api_key = article_data[0]
                        self.log_invalid_key(api_key)
                    elif article_data[1] == 'Error':
                        print(f"Statge 1 - Error was occurred while get summary\n: {article_data[0]}")
                    else:
                        writer.writerow(article_data[0:-1])
                        self.logger.log(f"Statge 1 - {sumarized_count}/{total} - {api_key} : {article_data[-1]}")
                        sumarized_count += 1
                        continue
                    articles.append([article_data[2], article_data[3], article_data[4]])
            pool.close()
            pool.join()
        end_t = time()
        self.logger.log(f'Stage 1 - {sumarized_count} articles were summarized in {end_t - start_t} seconds')

    def stage_1_save_db(self, csv_filename):
        self.logger.log(f'Stage 1 - loading summaries from {csv_filename}')
        start_t = time()
        data_list = []
        with open(csv_filename, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                if not row:
                    continue
                article = row[0]
                title = row[2]
                category = row[3]
                summary = row[4]
                day_score = row[5]
                day_reason = row[6]
                week_score = row[7]
                week_reason = row[8]
                month_score = row[9]
                month_reason = row[10]
                site_name = row[11]
                link = row[12]

                score = {
                    "day": {"score": day_score, "reason": day_reason},
                    "week": {"score": week_score, "reason": week_reason},
                    "month": {"score": month_score, "reason": month_reason}
                }
                data_list.append(
                    {"article": article,
                    "title": title,
                    "category": category,
                    "summary": summary,
                    "score": score,
                    "site_name": site_name,
                    "link": link}
                )
        end_t = time()
        self.logger.log(f'Stage 1 - loaded {len(data_list)} summaries in {end_t - start_t} seconds')
        self.logger.log(f'Stage 1 - saving summaries in db...')
        start_t = time()
        self.db["main_articles"].drop()
        new_collection = self.db['main_articles']
        result = new_collection.insert_many(data_list)
        print("Data inserted successfully. Inserted IDs:", result.inserted_ids)
        end_t = time()
        self.logger.log(f'Stage 1 - Saved {len(result.inserted_ids)} summaries in db...')

    def stage_2(self, stage1_csv: str, csv_filename: str, timeframe: str):
        os.remove(csv_filename) if os.path.exists(csv_filename) else None
        self.logger.log(f'Stage 2 - loading articles from {stage1_csv}')
        titles = []
        categories = set()
        total = 0
        start_t = time()
        with open(stage1_csv, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)

            for row in csv_reader:
                total += 1
                try:
                    if row[2] == '': continue
                    if timeframe == 'day' and row[5] == '': continue
                    if timeframe == 'week' and row[7] == '': continue
                    if timeframe == 'month' and row[9] == '': continue
                    score = 0
                    if timeframe == 'day':
                        score = int(remove_non_numbers_regex(row[5]))
                    elif timeframe == 'week':
                        score = int(remove_non_numbers_regex(row[7]))
                    elif timeframe == 'month':
                        score = int(remove_non_numbers_regex(row[9]))

                    article = row[0]
                    title = row[2]
                    category = row[3]  # Assuming the category is in the 4th column
                    summary = row[4]
                    categories.add(category)  # Add category to the set of unique categories

                    if any(item['title'] == title for item in titles):
                        print(title)
                        continue
                    titles.append({'article': article, 'title': title, 'score': score, 'category': category, 'summary': summary})
                except IndexError as err:
                    self.logger.log(f'Stage 2 - Error while loading articles: {err}')
                    pass
        end_t = time()
        self.logger.log(f'Stage 2 - articles were loaded from {stage1_csv} in {end_t - start_t} second')

        for category in categories:
            if category not in self.categories:
                continue
            category_titles = [title for title in titles if title['category'] == category]
            sorted_titles = sorted(category_titles, key=lambda x: x['score'], reverse=True)

            primaries = []
            for title in sorted_titles[0:20]:
                if title['title'] != '':
                    primaries.append(title['title'])
            secondaries = []
            for title in sorted_titles[20:270]:
                if title['title'] != '':
                    secondaries.append(title['title'])

            result = []
            try:
                result = self.stage_2_category(primaries=primaries, secondaries=secondaries)
                print(result[0])
                self.logger.log(f"Stage 2: {category} for {timeframe}-timeframe:\n {result[1]}")
            except Exception as er:
                error = str(traceback.print_exc())
                self.logger.log(f"Stage 2: Error while categorizing: {er} in {error}")
            try:
                data = json.loads(result[0])
                if data:
                    with open(csv_filename, "a", encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow([category, primaries, secondaries, data])
                else:
                    print('Error')
            except json.decoder.JSONDecodeError as err:
                self.logger.log(f"Stage 2: JSONDecode Error: {err} in {result}")

    def stage_2_category(self, primaries, secondaries):
        apikey = random.choice(self.apikeys)
        try:
            result = categorize(apikey=apikey, primaries=primaries, secondaries=secondaries)
            return result
        except InvalidRequestError as er:
            print(er)
        except AuthenticationError as er:
            self.log_invalid_key(apikey)
            self.logger.log(f"Stage 2: Invalid apikey: {apikey}")
            self.stage_2_category(primaries, secondaries)
        except RateLimitError as er:
            print(f"args: {er.args}\nparam: {er.code}, error: {er.error}, header: {er.headers}")
            if er.error['type'] == 'insufficient_quota':
                self.logger.log(f"Stage 2: Invalid apikey: {apikey}")
                self.stage_2_category(primaries, secondaries)
            # elif er.error['type'] == 'requests' and er.error['code'] == 'rate_limit_exceeded':
            elif er.error['code'] == 'rate_limit_exceeded':
                # check if remaining requests are not 0
                self.stage_2_category(primaries, secondaries)
                # if er.headers['x-ratelimit-remaining-requests'] == 0:
                # else:
                #     sleep(20)
                #     return stage_1_thread_handler( apikey, article, site_name, link,)
    
    def stage_2_save_db(self, csv_filename: str, collection: str):
        data_list = []
        with open(csv_filename, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if not row:
                    continue
                category = row[0]
                data = row[2]
                dictionary = eval(data)
                data_list.append({"category": category, "data": dictionary})

        self.db[collection].drop()
        new_collection = self.db[collection]

        result = new_collection.insert_many(data_list)
        self.logger.log(f"Stage 2 - data saved from {csv_filename} into {collection} collection")

def stage_1_thread_handler(
        apikey: str,
        article: str,
        site_name: str,
        link: str,
    ):
    items = [
        'Title:',
        'Category:',
        'Summary:',
        'Importance 1 day:',
        'Reasoning for 1 day score:',
        'Importance 1 week:',
        'Reasoning for 1 week score:',
        'Importance 1 month:',
        'Reasoning for 1 month score:',
    ]

    item_dict = {
        'Title:': None,
        'Category:': None,
        'Summary:': None,
        'Importance 1 day:': None,
        'Reasoning for 1 day score:': None,
        'Importance 1 week:': None,
        'Reasoning for 1 week score:': None,
        'Importance 1 month:': None,
        'Reasoning for 1 month score:': None
    }

    try:
        summary = summarize_article(apikey, article)
        for item in items:
            for line in summary[0].split('\n'):
                if line.startswith(item):
                    content = line[len(item):].strip()
                    item_dict[item] = content
                    break
        return [
            article,
            summary[0],
            item_dict['Title:'],
            item_dict['Category:'],
            item_dict['Summary:'],
            item_dict['Importance 1 day:'],
            item_dict['Reasoning for 1 day score:'],
            item_dict['Importance 1 week:'],
            item_dict['Reasoning for 1 week score:'],
            item_dict['Importance 1 month:'],
            item_dict['Reasoning for 1 month score:'],
            site_name,
            link,
            summary[1]
        ]
    except InvalidRequestError as er:
        return [er, 'Error', article, site_name, link]
    except RateLimitError as er:
        print(f"args: {er.args}\nparam: {er.code}, error: {er.error}, header: {er.headers}")
        if er.error['type'] == 'insufficient_quota':
            return [apikey, 'APIKey_Error', article, site_name, link]
        # elif er.error['type'] == 'requests' and er.error['code'] == 'rate_limit_exceeded':
        elif er.error['code'] == 'rate_limit_exceeded':
            # check if remaining requests are not 0
            return [er, 'Error', article, site_name, link]
            # if er.headers['x-ratelimit-remaining-requests'] == 0:
            # else:
            #     sleep(20)
            #     return stage_1_thread_handler( apikey, article, site_name, link,)
    except AuthenticationError as er:
        return [apikey, 'APIKey_Error', article, site_name, link]
    except Exception as er:
        # if 'Limit: 200 / day' in str(er):
        #     sleep(450)
        #     return stage_1_thread_handler( apikey, article, site_name, link,)
        # if 'Limit: 3 / min' in str(er):
        #     sleep(20)
        #     return stage_1_thread_handler( apikey, article, site_name, link,)
        # if 'Limit: 3 / min' in str(er) and 'Rate limit reached' in str(er):
        #     return stage_1_thread_handler(str, article, site_name, link)
        return [er, 'Error', article, site_name, link]