import requests
import time
import xml.etree.ElementTree as ET
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
import logging
import sys
import configparser

# load db credentials and auth token for bgg api
config = configparser.ConfigParser()
config.read('config.ini')
db = config['postgresql']
api = config['api']

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("script.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_user_collection(user):
    # own = 1, prevowned = 1, want = 1, wishlist = 1, wishlist_priority = 1.
    url = 'https://boardgamegeek.com/xmlapi2/collection?username=' + user + '&excludesubtype=boardgameexpansion&fortrade=0&wanttoplay=0&wanttobuy=0&preordered=0&stats=1'
    logging.info(url)
    fail = False
    t1 = time.time()
    i = 0

    # authorization
    header = {"Authorization": f"Bearer {api['auth_token']}"}

    # BGG API call
    while True:
        i += 1
        time1 = time.time()
        response = requests.get(url, headers=header)
        logging.info('Attempt ' + str(i+1))
        logging.info(str(response.status_code))

        # added delay between calls due to rate throttling
        time2 = time.time()
        diff = time2 - time1
        delay = max(0, 5-diff)
        time.sleep(delay)

        # break if successful
        if response.status_code == 200:
            break

        # fail API call if takes too many attempts
        if i >= 5:
            fail = True
            break

    t2 = time.time()
    logging.info('Approximate time: ' + str(t2-t1))

    # check if API call failed
    if fail:
        user = 0
        response = 0
        return user, response

    return user, response.content.decode('utf-8')

# just in case, mostly useless except for edge cases
def safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

# psql server connection
conn = psycopg2.connect(
    dbname=db['dbname'],
    user=db['user'],
    password=db['password'],
    host=db['host'],
    port=db['port']
)

df = sqlio.read_sql_query("SELECT * FROM userlist", conn)

cur = conn.cursor()
i = 0
count = 0

time1 = time.time()

processed = []

for index, row in df.iterrows():
    if row['processed'] == 0:
        # do API call
        user, response = get_user_collection(row['username'])

        if not isinstance(user, int):
            processed.append(row['username'])

            i += 1
            logging.info('Username ' + str(i) + '/561595, ' + str(round(i*100/561595, 2)))
            logging.info('Time elapsed: ' + str(time.time()-time1))

            # parse XML response
            try:
                root = ET.fromstring(response)
            except ET.ParseError:
                logging.warning(f"Skipping {row['username']}, XML parse failed")
                continue

            # prepare entry in db
            for item in root:
                username = row['username']
                objectid = safe_int(item.get('id', ''))
                collid = safe_int(item.get('collid', ''))
                name = item.find('name').text
                yearpublished = safe_int(item.find('yearpublished').text)
                own = item.find('status').get('own', '') == '1'
                prevowned = item.find('status').get('prevowned', '') == '1'
                want = item.find('status').get('want', '') == '1'
                wishlist = item.find('status').get('wishlist', '') == '1'
                wishlistpriority = safe_int(item.find('status').get('wishlistpriority', ''))
                numplays = safe_int(item.get('numplays', ''))

                # add data to db
                cur.execute("""
                        INSERT INTO boardgames
                        (username, objectid, collid, name, yearpublished, own, prevowned, want, wishlist, wishlistpriority, numplays)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                            (username, objectid, collid, name, yearpublished, own, prevowned, want, wishlist,
                             wishlistpriority, numplays)
                            )

        else:
            logging.warning(f"Skipping {row['username']}, API Call failed")

        count += 1
        # commits every 1000 rows written to server
        if count >= 1000:
            cur.executemany("UPDATE users SET processed=1 WHERE username=%s", [(u,) for u in processed])
            processed = []
            count = 0
            conn.commit()

cur.executemany("UPDATE users SET processed=1 WHERE username=%s", [(u,) for u in processed])
conn.commit()
cur.close()
conn.close()
# took a few months to collect the data on about half a million users
logging.info('Finished. Total time ' + str(time.time()-time1))