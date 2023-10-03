import os
import csv
import multiprocessing
from multiprocessing.pool import ApplyResult
from pymongo import MongoClient
from pymongo.database import Database
from dotenv import find_dotenv, load_dotenv
from summarize import summarize_article
from logger import Logger
from time import time, sleep
from openai.error import InvalidRequestError, RateLimitError, AuthenticationError
import random

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
            "Business and Finance"
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
        os.remove(csv_filename) if os.path.exists('result.csv') else None

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