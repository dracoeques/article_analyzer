import os
import csv
import multiprocessing
from multiprocessing.pool import ApplyResult
from pymongo import MongoClient
from pymongo.database import Database
from dotenv import find_dotenv, load_dotenv
from stages import summarize_article, categorize, extra_research, deep_research, impactul_news, prediction
from logger import Logger
from time import time, sleep
from openai.error import InvalidRequestError, RateLimitError, AuthenticationError
import random
from helpers import remove_non_numbers_regex
import json
import traceback
from datetime import date

# find and load .env file
load_dotenv(find_dotenv())

csv.field_size_limit(50000000)

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
        self.article_db = self.session["test"]
        self.collections = {
            0: "All categories",
            1: "lawandcrime",
            2: "web3",
            3: "entertainment",
            4: "sport",
            5: "artandfashion",
            6: "bizandfinance",
            7: "politics",
            8: "scienceandteche",
            9: "lifestyleandhealth",
            10: "gaming"
        }
        self.categories = [
            "Law and Crime",
            "Crypto/Web3",
            "Entertainment",
            "Sports",
            "Art and Fashion",
            "Business and Finance",
            "Politics",
            "Science and Technology",
            "Lifestyle and Health",
            "Gaming",
        ]
        with open('keys/keys.txt', 'r', encoding='utf-8') as keys_file:
            # Maximum 50 processes
            self.apikeys = [line.strip() for line in keys_file.readlines()[:50]]
    
    def log_invalid_key(self, apikey):
        # remove invallid apikey from valid list
        if apikey in self.apikeys:
            self.apikeys.remove(apikey)
            with open('keys/keys.txt', 'w', encoding='utf-8') as keys_file:
                for key in self.apikeys:
                    keys_file.write(key + '\n')

            # add apikey to invalid key file
            with open('keys/invalid_keys.txt', 'a', encoding='utf-8') as invalid_file:
                invalid_file.write(apikey + '\n')

    def stage_1(self, csv_filename: str, curDate: str):
        os.remove(csv_filename) if os.path.exists(csv_filename) else None

        start_t = time()
        # to test
        collection_name = [self.collections[i] for i in range(1, 11)]
        articles = []
        article_count = 0

        for idx, category in enumerate(collection_name):
            collection = self.article_db[category][curDate]
            rcategory = self.categories[idx]
            documents = collection.find()
            for document in documents:
                article_count += 1
                print(f"{article_count} : {rcategory}: {document['siteName']}, {document['link']}")
                articles.append([document['article'], document['siteName'], document['link'], rcategory])
        end_t = time()
        
        self.logger.log(f'Stage 1 - {len(articles)} articles uploaded in {end_t - start_t} seconds, start processing...')
        
        start_t = time()
        article_count = 0

        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'baseData',
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
            csvfile.close()
        sumarized_count = 0
        total = len(articles)
        while len(articles) != 0:
            pool = multiprocessing.Pool(processes=min(len(self.apikeys), 50))
            results: list[ApplyResult] = []

            for i, (article, site_name, link, rCategory) in enumerate(articles):
                api_key = self.apikeys[i % len(self.apikeys)]  # Use a different API key for each process
                if len(self.apikeys) > len(articles):
                    api_key = random.choice(self.apikeys)
                results.append(pool.apply_async(stage_1_thread_handler, (api_key, article, site_name, link, rCategory)))
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
                        self.logger.log(f"Statge 1 - {sumarized_count}/{total} : {article_data[-1]}")
                        sumarized_count += 1
                        continue
                    articles.append([article_data[2], article_data[3], article_data[4]])
            pool.close()
            pool.join()
        end_t = time()
        self.logger.log(f'Stage 1 - {sumarized_count} articles were summarized in {end_t - start_t} seconds')

    def stage_1_save_db(self, csv_filename, curDate: str):
        self.logger.log(f'Stage 1 - loading summaries from {csv_filename}')
        start_t = time()
        data_list = []
        with open(csv_filename, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                if not row:
                    continue
                baseData = json.loads(row[0])
                article = baseData['article']
                title = row[1]
                category = row[2]
                summary = row[3]
                day_score = row[4]
                day_reason = row[5]
                week_score = row[6]
                week_reason = row[7]
                month_score = row[8]
                month_reason = row[9]
                site_name = row[10]
                link = row[11]

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
        new_collection = self.db['analyzed_articles'][curDate]
        new_collection.drop()
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
                    if row[1] == '': continue
                    if timeframe == 'day' and row[4] == '': continue
                    if timeframe == 'week' and row[6] == '': continue
                    if timeframe == 'month' and row[8] == '': continue
                    score = 0
                    if timeframe == 'day':
                        score = int(remove_non_numbers_regex(row[4]))
                    elif timeframe == 'week':
                        score = int(remove_non_numbers_regex(row[6]))
                    elif timeframe == 'month':
                        score = int(remove_non_numbers_regex(row[8]))

                    article = json.loads(row[0])['article']
                    title = row[1]
                    category = row[2]  # Assuming the category is in the 4th column
                    summary = row[3]
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
                if title['title'] != '' and title['title'] not in primaries:
                    primaries.append(title['title'])
            secondaries = []
            for title in sorted_titles[20:270]:
                if title['title'] != '' and title['title'] not in primaries and title['title'] not in secondaries:
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
                    self.logger.log(f"Stage 2: {len(data)}")
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
            return self.stage_2_category(primaries, secondaries)
        except RateLimitError as er:
            print(f"args: {er.args}\nparam: {er.code}, error: {er.error}, header: {er.headers}")
            if er.error['type'] == 'insufficient_quota':
                self.logger.log(f"Stage 2: Invalid apikey: {apikey}")
                self.log_invalid_key(apikey)
                return self.stage_2_category(primaries, secondaries)
            # elif er.error['type'] == 'requests' and er.error['code'] == 'rate_limit_exceeded':
            elif er.error['code'] == 'rate_limit_exceeded':
                # check if remaining requests are not 0
                return self.stage_2_category(primaries, secondaries)
                # if er.headers['x-ratelimit-remaining-requests'] == 0:
                # else:
                #     sleep(20)
                #     return stage_1_thread_handler( apikey, article, site_name, link,)
    
    def stage_2_save_db(self, csv_filename: str, collection: str, curDate: str):
        data_list = []
        with open(csv_filename, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if not row:
                    continue
                category = row[0]
                data = row[3]
                dictionary = eval(data)
                data_list.append({"category": category, "data": dictionary})

        new_collection = self.db[collection][curDate]
        new_collection.drop()

        result = new_collection.insert_many(data_list)
        self.logger.log(f"Stage 2 - data saved from {csv_filename} into {collection} collection")

    def stage_3(self, category_csv, summary_csv, csv_filename):
        os.remove(csv_filename) if os.path.exists(csv_filename) else None
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'category',
                'topic',
                'research',
                'articles'
            ])
        toprompts = []
        summaries = []
        self.logger.log(f"Stage 3 - Preparing article and topic datas")

        with open(summary_csv, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                summaries.append({
                    "category": row[2],
                    "title": row[1],
                    "summary": row[3],
                    "content": json.loads(row[0])['article'],
                })
        total = 0
        with open(category_csv, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if not row:
                    continue
                category = row[0]
                topics =  eval(row[3])
                print(len(topics))
                if len(topics) < 20:
                    print(category, topics)
                for topic in topics:
                    total += 1
                    primary = topic["Primary"]
                    secondary = topic["Secondary"]
                    articles = []
                    for title in secondary:
                        summary = ""
                        content = ""
                        for ele in summaries:
                            if ele["title"] == title:
                                summary = ele["summary"]
                                content = ele["content"]
                                break
                        articles.append({
                            "title": title,
                            "summary": summary,
                            "content": content,
                        })
                    if len(articles) == 0: continue
                    toprompt = {
                        "topic": primary,
                        "category": category,
                        "articles": articles,
                    }
                    toprompts.append(toprompt)
        self.logger.log(f"Stage 3 - Prepared {len(toprompts)} article and topic data")
        self.logger.log(f"Stage 3 - Start extra research...")
        
        researched_count = 0
        total = len(toprompts)
        start_t = time()
        while len(toprompts) != 0:
            pool = multiprocessing.Pool(processes=min(len(self.apikeys), 50))
            results: list[ApplyResult] = []

            for i, toprompt in enumerate(toprompts):
                topic = toprompt["topic"]
                category = toprompt["category"]
                articles = toprompt["articles"]
                api_key = self.apikeys[i % len(self.apikeys)]  # Use a different API key for each process
                if len(self.apikeys) > len(toprompts):
                    api_key = random.choice(self.apikeys)
                results.append(pool.apply_async(stage_3_thread_handler, (api_key, category, topic, articles,)))
            toprompts = []
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                for result in results:
                    article_data = result.get()
                    if article_data[1] == 'APIKey_Error':
                        api_key = article_data[0]
                        self.log_invalid_key(api_key)
                    elif article_data[1] == 'Error':
                        print(f"Statge 3 - Error was occurred in extra research\n: {article_data[0]}")
                    else:
                        writer.writerow(article_data[0:-1])
                        researched_count += 1
                        self.logger.log(f"Statge 3 - {researched_count}/{total} : {article_data[-1]}")
                        continue
                    toprompt = {
                        "topic": article_data[2],
                        "category": article_data[3],
                        "articles": article_data[4],
                    }
                    toprompts.append(toprompt)
            pool.close()
            pool.join()
        end_t = time()
        self.logger.log(f'Stage 3 - {researched_count} articles were extra researched in {end_t - start_t} seconds')

    def stage_3_save_db(self, csv_filename: str, collection: str, curDate: str):
        data_list = []
        categories = set()
        researches = []
        with open(csv_filename, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                if not row:
                    continue
                categories.add(row[0])
                researches.append({
                    "category": row[0],
                    "topic": row[1],
                    "research": eval(row[2]),
                })
        for category in categories:
            data_list.append({
                "category": category,
                "data": [{
                    "topic": item["topic"],
                    "research": item["research"]
                } for item in researches if item["category"] == category]
            })
        new_collection = self.db[collection][curDate]
        new_collection.drop()
        result = new_collection.insert_many(data_list)
        self.logger.log(f"Stage 3 - data saved from {csv_filename} into {collection} collection")
    
    def stage_4(self, stage3_csv: str, csv_filename: str):
        os.remove(csv_filename) if os.path.exists(csv_filename) else None
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'category',
                'topic',
                'background',
                'deep_research'
                'articles'
            ])

        data = []
        categories = set()
        start_t = time()
        self.logger.log(f"Stage 4 - Loading data from {stage3_csv}...")
        with open(stage3_csv, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                if not row:
                    continue
                categories.add(row[0])
                data.append({
                    "category": row[0],
                    "topic": row[1],
                    "research": eval(row[2]),
                    "articles": eval(row[3])
                })
        end_t = time()
        self.logger.log(f"Stage 4 - loaded {len(data)} data from {stage3_csv} in {end_t - start_t} seconds")

        total = len(data)
        start_t = time()
        researched = 0
        while len(data) != 0:
            pool = multiprocessing.Pool(processes=min(len(self.apikeys), 50))
            results: list[ApplyResult] = []

            for i, item in enumerate(data):
                topic = item["topic"]
                category = item["category"]
                research = item["research"]
                articles = item["articles"]
                api_key = self.apikeys[i % len(self.apikeys)]  # Use a different API key for each process
                if len(self.apikeys) > len(data):
                    api_key = random.choice(self.apikeys)
                results.append(pool.apply_async(stage_4_thread_handler, (api_key, category, topic, research, articles,)))
            data = []
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                for result in results:
                    article_data = result.get()
                    if article_data[1] == 'APIKey_Error':
                        api_key = article_data[0]
                        self.log_invalid_key(api_key)
                    elif article_data[1] == 'Error':
                        print(f"Statge 4 - Error was occurred in deep research\n: {article_data[0]}")
                        continue
                    else:
                        writer.writerow([article_data[0], article_data[1], article_data[2], article_data[3], str(article_data[4])])
                        researched += 1
                        self.logger.log(f"Statge 4 - {researched}/{total} : {article_data[-1]}")
                        continue
                    data.append({
                        "category": article_data[2],
                        "topic": article_data[3],
                        "research": article_data[4],
                        "articles": article_data[5]
                    })
            pool.close()
            pool.join()
        end_t = time()
        self.logger.log(f'Stage 4 - {researched} articles were extra researched in {end_t - start_t} seconds')
    
    def stage_4_save_db(self, csv_filename: str, collection: str, curDate: str):
        data_list = []
        categories = set()
        dresearches = []
        with open(csv_filename, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                if not row:
                    continue
                categories.add(row[0])
                dresearches.append({
                    "category": row[0],
                    "topic": row[1],
                    "deep_research": eval(row[3]),
                })
        for category in categories:
            print(row[3])
            data_list.append({
                "category": category,
                "data": [{
                    "topic": item["topic"],
                    "deep_research": item["deep_research"]
                } for item in dresearches if item["category"] == category]
            })
        new_collection = self.db[collection][curDate]
        new_collection.drop()
        result = new_collection.insert_many(data_list)
        self.logger.log(f"Stage 4 - data saved from {csv_filename} into {collection} collection")
    
    def stage_5(self, stage1_csv, csv_filename, timeframe):
        self.logger.log(f'Stage 5 - loading articles from {stage1_csv}')
        articles = []
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

                    # article = row[0]
                    title = row[2]
                    category = row[3]  # Assuming the category is in the 4th column
                    summary = row[4]
                    categories.add(category)  # Add category to the set of unique categories

                    if any(item['title'] == title for item in articles):
                        continue
                    articles.append({'title': title, 'score': score, 'category': category, 'summary': summary})
                except IndexError as err:
                    self.logger.log(f'Stage 2 - Error while loading articles: {err}')
                    pass
        end_t = time()
        self.logger.log(f'Stage 5 - articles were loaded from {stage1_csv} in {end_t - start_t} second')
        top30 = []
        start_t = time()
        for category in categories:
            if category not in self.categories:
                continue
            category_articles = [article for article in articles if article['category'] == category]
            sorted_articles = sorted(category_articles, key=lambda x: x['score'], reverse=True)
            for tmp in sorted_articles[0:3]:
                top30.append(tmp)
        end_t = time()
        self.logger.log(f'Stage 5 - articles were sorted in {end_t - start_t} second')
        start_t = time()
        result = []
        try:
            result = self.stage_5_impactful_news(top30)
        except Exception as er:
            error = str(traceback.print_exc())
            self.logger.log(f"Stage 5: Error : {er} in {error}")
        self.logger.log(f'Stage 5 - {result[1]}')
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["no", "title", "explanation"])
            print(result[0])
            tops: list = eval(result[0])
            print(tops)
            for i, top in enumerate(tops):
                writer.writerow([i+1, top['title'], top['explanation']])

        end_t = time()
        self.logger.log(f'Stage 5 - got the result in {end_t - start_t} second')
    
    def stage_5_impactful_news(self, articles):
        apikey = random.choice(self.apikeys)
        try:
            result = impactul_news(apikey=apikey, articles=articles)
            return result
        except InvalidRequestError as er:
            result = self.stage_5_impactful_news(articles)
            return result
        except AuthenticationError as er:
            self.log_invalid_key(apikey)
            self.logger.log(f"Stage 5: Invalid apikey: {apikey}")
            return self.stage_5_impactful_news(articles)
        except RateLimitError as er:
            print(f"args: {er.args}\nparam: {er.code}, error: {er.error}, header: {er.headers}")
            if er.error['type'] == 'insufficient_quota':
                self.logger.log(f"Stage 5: Invalid apikey: {apikey}")
                self.log_invalid_key(apikey)
                return self.stage_5_impactful_news(articles)
            elif er.error['code'] == 'rate_limit_exceeded':
                return self.stage_5_impactful_news(articles)
    
    def stage_5_save_db(self, csv_filename: str, collection: str, curDate: str):
        data_list = []
        with open(csv_filename, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                if not row:
                    continue
                title = row[1]
                explanation = row[2]
                data_list.append({"title": title, "explanation": explanation})

        new_collection = self.db[collection][curDate]
        new_collection.drop()
        result = new_collection.insert_many(data_list)
        self.logger.log(f"Stage 5 - data saved from {csv_filename} into {collection} collection")
    
    def stage_6(self, stage4_csv: str, csv_filename: str, timeframe: str):
        self.logger.log(f'Stage 6 - loading topics from {stage4_csv}')
        topics = []
        data_list = []
        categories = set()
        start_t = time()
        with open(stage4_csv, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                if not row:
                    continue
                categories.add(row[0])
                deep_research = eval(row[3])
                topics.append({
                    "category": row[0],
                    "topic": row[1],
                    "prediction": f"Description: {deep_research['1 day timeframe']['Most likely']['Description']}\nExplanation: {deep_research['1 day timeframe']['Most likely']['Explanation']}",
                })
        for category in categories:
            if category not in self.categories:
                continue
            data_list.append({
                "category": category,
                "data": [{
                    "topic": topic["topic"],
                    "prediction": topic["prediction"]
                } for topic in topics if topic["category"] == category]
            })
        data_list.append({
            "category": "at_glance",
            "data": [{
                "topic": topic["topic"],
                "prediction": topic["prediction"]
            } for topic in topics]
        })
        end_t = time()
        self.logger.log(f'Stage 6 - topics were loaded from {stage4_csv} in {end_t - start_t} second')
        start_t = time()
        results = []
        try:
            for i, data in enumerate(data_list):
                result = self.stage_6_prediction(data, timeframe)
                results.append([result[0], eval(result[1][0])])
                self.logger.log(f'Stage 6 - {i+1}/{len(data_list)} - {result[1][1]}')
        except Exception as er:
            error = str(traceback.print_exc())
            self.logger.log(f"Stage 6: Error : {er} in {error}")
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["category", "prediction"])
            for result in results:
                writer.writerow(result)

        end_t = time()
        self.logger.log(f'Stage 6 - got the result in {end_t - start_t} second')

    def stage_6_prediction(self, topics, timeframe):
        apikey = random.choice(self.apikeys)
        try:
            result = prediction(apikey, topics["data"], topics["category"], timeframe)
            return [topics["category"], result]
        except InvalidRequestError as er:
            result = self.stage_6_prediction(topics, timeframe)
            return result
        except AuthenticationError as er:
            self.log_invalid_key(apikey)
            self.logger.log(f"Stage 6: Invalid apikey: {apikey}")
            return self.stage_6_prediction(topics, timeframe)
        except RateLimitError as er:
            print(f"args: {er.args}\nparam: {er.code}, error: {er.error}, header: {er.headers}")
            if er.error['type'] == 'insufficient_quota':
                self.logger.log(f"Stage 6: Invalid apikey: {apikey}")
                self.log_invalid_key(apikey)
                return self.stage_6_prediction(topics, timeframe)
            elif er.error['code'] == 'rate_limit_exceeded':
                return self.stage_6_prediction(topics, timeframe)
    
    def stage_6_save_db(self, csv_filename: str, collection: str, curDate: str):
        data_list = []
        with open(csv_filename, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                if not row:
                    continue
                data_list.append({"category": row[0], "prediction": eval(row[1])})

        new_collection = self.db[collection][curDate]
        new_collection.drop()
        result = new_collection.insert_many(data_list)
        self.logger.log(f"Stage 6 - data saved from {csv_filename} into {collection} collection")

def stage_1_thread_handler(
        apikey: str,
        article: str,
        site_name: str,
        link: str,
        rCategory: str,
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
        'Category:': rCategory,
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
        baseData = {
            'article': article,
            'result': summary[0]
        }
        return [
            json.dumps(baseData),            
            item_dict['Title:'],
            rCategory,
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
        return [er, 'Error', article, site_name, link, rCategory]
    except RateLimitError as er:
        print(f"args: {er.args}\nparam: {er.code}, error: {er.error}, header: {er.headers}")
        if er.error['type'] == 'insufficient_quota':
            return [apikey, 'APIKey_Error', article, site_name, link, rCategory]
        # elif er.error['type'] == 'requests' and er.error['code'] == 'rate_limit_exceeded':
        elif er.error['code'] == 'rate_limit_exceeded':
            # check if remaining requests are not 0
            return [er, 'Error', article, site_name, link, rCategory]
            # if er.headers['x-ratelimit-remaining-requests'] == 0:
            # else:
            #     sleep(20)
            #     return stage_1_thread_handler( apikey, article, site_name, link,)
    except AuthenticationError as er:
        return [apikey, 'APIKey_Error', article, site_name, link, rCategory]
    except Exception as er:
        # if 'Limit: 200 / day' in str(er):
        #     sleep(450)
        #     return stage_1_thread_handler( apikey, article, site_name, link,)
        # if 'Limit: 3 / min' in str(er):
        #     sleep(20)
        #     return stage_1_thread_handler( apikey, article, site_name, link,)
        # if 'Limit: 3 / min' in str(er) and 'Rate limit reached' in str(er):
        #     return stage_1_thread_handler(str, article, site_name, link)
        return [er, 'Error', article, site_name, link, rCategory]

def stage_3_thread_handler(
        apikey: str,
        category: str,
        topic: str,
        articles: dict,
    ):
    try:
        contents = [article['content'] for article in articles]
        summary = extra_research(apikey, contents)
        return [category, topic, json.loads(summary[0].replace("\n", " ")), articles,  summary[1]]
    except InvalidRequestError as er:
        return [er, 'Error', category, topic, articles]
    except RateLimitError as er:
        print(f"args: {er.args}\nparam: {er.code}, error: {er.error}, header: {er.headers}")
        if er.error['type'] == 'insufficient_quota':
            return [apikey, 'APIKey_Error', category, topic, articles]
        elif er.error['code'] == 'rate_limit_exceeded':
            return [er, 'Error', category, topic, articles]
    except AuthenticationError as er:
        return [apikey, 'APIKey_Error', category, topic, articles]
    except Exception as er:
        error = str(traceback.print_exc())
        print(error)
        return [er, 'Error', category, topic, articles]

def stage_4_thread_handler(
        apikey: str,
        category: str,
        topic: str,
        research: dict,
        articles: list,
    ):
    try:
        summary = deep_research(apikey, articles, background=research)
        eval(summary[0])
        return [category, topic, research, summary[0], articles,  summary[1]]
    except InvalidRequestError as er:
        return [er, 'Error', category, topic, research, articles]
    except RateLimitError as er:
        print(f"args: {er.args}\nparam: {er.code}, error: {er.error}, header: {er.headers}")
        if er.error['type'] == 'insufficient_quota':
            return [apikey, 'APIKey_Error', category, topic, research, articles]
        elif er.error['code'] == 'rate_limit_exceeded':
            return [er, 'Error', category, topic, research, articles]
    except AuthenticationError as er:
        return [apikey, 'APIKey_Error', category, topic, research, articles]
    except Exception as er:
        error = str(traceback.print_exc())
        print(error)
        return [er, 'Error', category, topic, research, articles]
